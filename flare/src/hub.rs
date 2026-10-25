//! Download prepared packages from hardware-specific Hugging Face revisions.

use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

use hf_hub::{
    Cache, Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};

use crate::cli::HubOptions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn is_repository(path: &Path) -> bool {
    !path.exists()
        && !path.is_absolute()
        && !path.starts_with(".")
        && !path.starts_with("..")
        && path.to_str().is_some_and(|text| {
            !text.contains('\\') && !text.ends_with(".safetensors") && repository(text).is_ok()
        })
}

pub(crate) fn repository(text: &str) -> Result<(&str, Option<&str>)> {
    let (id, revision) = match text.split_once('@') {
        Some((id, revision)) if !revision.is_empty() => (id, Some(revision)),
        Some(_) => return Err("Hugging Face revision cannot be empty".into()),
        None => (text, None),
    };
    let parts: Vec<_> = id.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
    {
        return Err("expected a Hugging Face repository as ORG/NAME[@REVISION]".into());
    }
    if let Some(revision) = revision {
        validate_relative(revision)?;
    }
    Ok((id, revision))
}

fn validate_relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.contains(['\\', '\0'])
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.contains(':'))
    {
        return Err(format!("invalid repository-relative path or revision: {value:?}").into());
    }
    Ok(())
}

fn is_companion(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".safetensors") else {
        return false;
    };
    let stem = stem.rsplit('/').next().unwrap_or(stem);
    if stem.len() == 64
        && stem
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return true;
    }
    let Some((_, numbers)) = stem.rsplit_once("-data-") else {
        return false;
    };
    let bytes = numbers.as_bytes();
    bytes.len() == 8
        && &bytes[2..6] == b"-of-"
        && bytes[..2].iter().chain(&bytes[6..]).all(u8::is_ascii_digit)
}

fn choose_file(files: impl IntoIterator<Item = String>, requested: Option<&str>) -> Result<String> {
    if let Some(filename) = requested {
        validate_relative(filename)?;
        if !filename.ends_with(".safetensors") || is_companion(filename) {
            return Err("--filename must name a prepared .safetensors package".into());
        }
        return Ok(filename.to_owned());
    }
    let mut candidates: Vec<_> = files
        .into_iter()
        .filter(|file| file.ends_with(".safetensors") && !is_companion(file))
        .filter(|file| is_canonical_primary(file))
        .collect();
    candidates.sort();
    match candidates.as_slice() {
        [filename] => {
            validate_relative(filename)?;
            Ok(filename.clone())
        }
        [] => Err("revision contains no .safetensors package; choose a revision containing prepared Flare packages".into()),
        _ => Err(format!("revision contains several .safetensors files; select one with --filename:\n  {}", candidates.join("\n  ")).into()),
    }
}

pub fn run_download(args: crate::cli::Download) -> Result<()> {
    crate::environment::configure(&args.gpus)?;
    let entry = download(
        &args.repository,
        &args.hub,
        args.gpus.len(),
        &args.publisher_keys,
        args.trust_store.as_deref(),
        &[],
    )?;
    println!("{}", entry.directory().display());
    Ok(())
}

pub fn download(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    let access_token = if options.offline {
        None
    } else {
        token(options)?
    };
    download_with_token(
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
    )
}

pub(crate) fn download_with_token(
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    download_with_token_at(
        "https://huggingface.co",
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_with_token_at(
    endpoint: &str,
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    download_for_model_at(
        endpoint,
        source,
        options,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_for_model_at(
    endpoint: &str,
    source: &str,
    options: &HubOptions,
    gpu_count: usize,
    access_token: Option<String>,
    publisher_keys: &[std::path::PathBuf],
    trust_store: Option<&std::path::Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
    expected_model: Option<&str>,
) -> Result<flare_helios::package::cache::Entry> {
    let (id, given_revision) = repository(source)?;
    let revision = match options.revision.as_deref().or(given_revision) {
        Some(revision) => revision.to_owned(),
        None => crate::hardware::detect_isolated(gpu_count)?,
    };
    validate_relative(&revision)?;
    if let Some(filename) = &options.filename {
        choose_file([], Some(filename))?;
    }
    let model_root = crate::cache_support::model_root(options.model_cache.as_deref())
        .map_err(|error| error.to_string())?;
    eprintln!("Prepared-model cache: {}", model_root.display());
    if options.offline {
        let (entry, expected_envelope) =
            offline_entry_with_expected(&model_root, id, &revision, options, gpu_count)?;
        let primary = entry_primary(&entry)?;
        let resolved = crate::trust::resolve_identity_with(
            &primary,
            publisher_keys,
            supplied_profiles,
            options,
            trust_store,
        )?;
        verify_offline_binding(
            &entry,
            expected_envelope,
            id,
            &revision,
            &resolved.profiles,
        )?;
        verify_offline_entry(&entry, options, gpu_count, &resolved.profiles)?;
        flare_helios::package::cache::Root::open(&model_root)?.publish_entry(&entry)?;
        return Ok(entry);
    }
    // Prepared package payloads are fetched into one operation-owned staging
    // tree. Concurrent acquisitions never share a deletable tree, and the tree
    // disappears when this operation ends. `--cache-dir` selects the parent for
    // that transport staging, not a permanent prepared-model cache.
    let report = AcquisitionReport::new();
    let transport = TransportStaging::new(&model_root, options.cache_dir.as_deref())?;
    let cache = Cache::new(transport.path().to_owned());
    let repo = Repo::with_revision(id.to_owned(), RepoType::Model, revision.clone());
    eprintln!("Package source: {id}@{revision}");
    let path = {
        let api = ApiBuilder::from_cache(cache.clone())
            .with_token(access_token)
            .with_endpoint(endpoint.into())
            .with_progress(std::io::stderr().is_terminal())
            .with_retries(3)
            .build()?;
        let (primary, pinned) = fetch_primary_with_progress(
            &api,
            &cache,
            &repo,
            options.filename.as_deref(),
            Some(&report.progress),
        )?;
        let filename = primary
            .strip_prefix(snapshot(&cache, &pinned, pinned.revision()))?
            .to_str()
            .ok_or("package filename is not UTF-8")?
            .to_owned();
        // The shared resolver supplies remembered and explicit profiles plus the
        // established identity; download never accepts terms or decodes executables.
        let resolved = crate::trust::resolve_identity_with(
            &primary,
            publisher_keys,
            supplied_profiles,
            options,
            trust_store,
        )?;
        let publishers = resolved.profiles;
        let legacy = flare_helios::package::conversion::form(&primary)
            .map_err(|error| error.to_string())?
            .is_none();
        if let Some(expected) = expected_model.filter(|_| !legacy) {
            use flare_helios::package::{cache, protection, reader};
            let metadata = reader::Metadata::open(&primary)?;
            let catalog = protection::verify(&metadata.envelope, &publishers, protection::Selection::Catalog)?;
            let identity = catalog.package().model_identity.as_str();
            let model = cache::model_key(if identity.is_empty() { id } else { identity })
                .map_err(|error| error.to_string())?;
            if model != expected {
                return Err("the Hugging Face update names a different model; the selected version is unchanged".into());
            }
        }
        let entry = if legacy {
            match legacy_warm_entry(
                &primary,
                options,
                &id,
                &revision,
                pinned.revision(),
                &filename,
                gpu_count,
                &publishers,
                &model_root,
            )? {
                Some(entry) => entry,
                None => {
                    for name in dependency_files(&primary, options, gpu_count, &publishers)? {
                        fetch_dependency_with_progress(
                            &api,
                            &cache,
                            &pinned,
                            &primary,
                            &name,
                            Some(&report.progress),
                        )?;
                    }
                    install_ready(&id, &primary, options, &model_root, gpu_count, &publishers)?
                }
            }
        } else {
            match plan_payload_acquisition(
                &primary,
                options,
                &id,
                &revision,
                pinned.revision(),
                &filename,
                gpu_count,
                &publishers,
                &model_root,
                Some(&report.progress),
            )? {
                PayloadAcquisition::Ready(entry) => entry,
                PayloadAcquisition::Planned { plan, staging } => {
                    let names = plan.remote_files();
                    for name in &names {
                        fetch_dependency_with_progress(
                            &api,
                            &cache,
                            &pinned,
                            &primary,
                            name,
                            Some(&report.progress),
                        )?;
                    }
                    finish_payload_acquisition(&primary, &names, plan, staging, &model_root)?
                }
            }
        };
        if expected_model.is_some_and(|model| model != entry.model()) {
            return Err("the Hugging Face update names a different model; the selected version is unchanged".into());
        }
        let expected_envelope =
            authenticated_source_envelope(&primary, options, gpu_count, &publishers)?;
        remember_download(
            &model_root,
            id,
            &revision,
            pinned.revision(),
            &filename,
            options,
            gpu_count,
            &entry,
            expected_envelope,
        )?;
        // The default switches only after the source mapping is durable; a failed
        // source recording must not leave a different model version selected.
        flare_helios::package::cache::Root::open(&model_root)?.publish_entry(&entry)?;
        drop(transport);
        entry
    };
    let mut report = report;
    report.success = true;
    Ok(path)
}

struct AcquisitionReport {
    progress: flare_helios::package::progress::AcquisitionProgress,
    success: bool,
}

impl AcquisitionReport {
    fn new() -> Self {
        Self {
            progress: flare_helios::package::progress::AcquisitionProgress::new(),
            success: false,
        }
    }
}

impl Drop for AcquisitionReport {
    fn drop(&mut self) {
        let snapshot = self.progress.snapshot();
        let state = if self.success { "completed" } else { "failed" };
        eprintln!(
            "Acquisition {state}: network {} bytes, local source {} bytes, materialized {} bytes, verification {} bytes, elapsed {:.3}s",
            snapshot.network_bytes,
            snapshot.local_source_bytes,
            snapshot.materialized_bytes(),
            snapshot.verification_bytes,
            snapshot.elapsed.as_secs_f64()
        );
    }
}

/// One acquisition's transport staging under the shared cache's transaction
/// area. Unique per operation; hf-hub retries and resumes within this tree, and
/// dropping it removes the tree. A crashed process leaves an abandoned staging
/// transaction that the cache collector recognizes; it is not resumed by a new
/// invocation.
enum TransportStaging {
    Explicit(tempfile::TempDir),
    Cache(flare_helios::package::cache::Staging),
}

impl TransportStaging {
    fn new(model_root: &Path, explicit: Option<&Path>) -> Result<Self> {
        match explicit {
            Some(parent) => {
                std::fs::create_dir_all(parent)?;
                Ok(Self::Explicit(
                    tempfile::Builder::new()
                        .prefix(".flare-hub-staging-")
                        .tempdir_in(parent)?,
                ))
            }
            None => {
                let root = flare_helios::package::cache::Root::open(model_root)?;
                Ok(Self::Cache(root.stage_named("hub-transfer", "transfer")?))
            }
        }
    }
    fn path(&self) -> &Path {
        match self {
            Self::Explicit(directory) => directory.path(),
            Self::Cache(staging) => staging.directory(),
        }
    }
}

fn source_selector(
    repository: &str,
    revision: &str,
    filename: Option<&str>,
    sizes: &[u32],
) -> String {
    serde_json::json!(["huggingface", repository, revision, filename, sizes]).to_string()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HubOrigin {
    repository: String,
    revision: String,
    commit: String,
    filename: Option<String>,
    sizes: Vec<u32>,
    expected_envelope: [u8; 32],
}

impl HubOrigin {
    fn from_reference(reference: &flare_helios::package::cache::SourceRef) -> Result<Self> {
        let (provider, repository_id, revision, filename, sizes): (
            String,
            String,
            String,
            Option<String>,
            Vec<u32>,
        ) = serde_json::from_str(reference.selector())?;
        if provider != "huggingface" || repository(&repository_id)?.1.is_some() {
            return Err("model origin is not a Hugging Face repository".into());
        }
        validate_relative(&revision)?;
        if let Some(filename) = &filename {
            choose_file([], Some(filename))?;
        }
        let expected_envelope = reference
            .expected_envelope()
            .ok_or_else(|| unbound_source(&repository_id, &revision))?;
        if sizes.is_empty() || sizes.contains(&0) {
            return Err("model origin has no concrete tensor-parallel selection".into());
        }
        Ok(Self {
            repository: repository_id,
            revision,
            commit: reference.revision().into(),
            filename,
            sizes,
            expected_envelope,
        })
    }
}

fn entry_origin(
    store: &flare_helios::package::cache::Root,
    entry: &flare_helios::package::cache::Entry,
) -> Result<Option<HubOrigin>> {
    if let Some(reference) = store.entry_source(entry)? {
        return HubOrigin::from_reference(&reference).map(Some);
    }
    // Source selectors can recover an origin only when they identify one acquisition unambiguously.
    let mut origins = std::collections::BTreeSet::new();
    for reference in store.entry_source_refs(entry)? {
        let value: serde_json::Value = match serde_json::from_str(reference.selector()) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get(0).and_then(serde_json::Value::as_str) != Some("huggingface") {
            continue;
        }
        let fields: (String, String, String, Option<String>, Vec<u32>) =
            serde_json::from_value(value)?;
        if fields.4.is_empty() || fields.3.is_none() {
            continue;
        }
        origins.insert(HubOrigin::from_reference(&reference)?);
    }
    if origins
        .iter()
        .any(|origin| origin.revision != origin.commit)
    {
        origins.retain(|origin| origin.revision != origin.commit);
    }
    if origins.len() > 1 {
        return Err("model version has multiple recorded Hugging Face origins; download the intended source explicitly to record its origin".into());
    }
    let origin = origins.into_iter().next();
    if let Some(origin) = &origin {
        store.publish_entry_source(
            &source_selector(
                &origin.repository,
                &origin.revision,
                origin.filename.as_deref(),
                &origin.sizes,
            ),
            &origin.commit,
            entry,
            origin.expected_envelope,
        )?;
    }
    Ok(origin)
}

pub(crate) fn refresh_model(
    store: &flare_helios::package::cache::Root,
    entry: flare_helios::package::cache::Entry,
    pinned: bool,
    args: &crate::cli::Run,
) -> Result<(flare_helios::package::cache::Entry, Option<String>)> {
    let entry = refresh_model_at(
        "https://huggingface.co",
        store,
        entry,
        pinned,
        &args.hub,
        args.gpus.len().max(1),
        &args.publisher_keys,
        args.trust_store.as_deref(),
        &[],
    )?;
    let repository = if pinned || args.hub.offline {
        None
    } else {
        store.entry_source(&entry)?.map(|reference| {
            HubOrigin::from_reference(&reference).map(|origin| origin.repository)
        }).transpose()?
    };
    Ok((entry, repository))
}

#[allow(clippy::too_many_arguments)]
fn refresh_model_at(
    endpoint: &str,
    store: &flare_helios::package::cache::Root,
    entry: flare_helios::package::cache::Entry,
    pinned: bool,
    options: &HubOptions,
    gpu_count: usize,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    supplied_profiles: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    if pinned {
        if options.revision.is_some() || options.filename.is_some() {
            return Err("a local model version is pinned; use models download to select a Hugging Face revision".into());
        }
        return Ok(entry);
    }
    if options.offline && options.revision.is_none() && options.filename.is_none() {
        return Ok(entry);
    }
    let Some(origin) = entry_origin(store, &entry)? else {
        if options.revision.is_some() || options.filename.is_some() {
            return Err("this local model has no Hugging Face origin; download the intended repository explicitly".into());
        }
        return Ok(entry);
    };
    if options.offline {
        if options
            .revision
            .as_ref()
            .is_some_and(|revision| revision != &origin.revision && revision != &origin.commit)
            || options
                .filename
                .as_ref()
                .is_some_and(|filename| Some(filename) != origin.filename.as_ref())
        {
            return Err(
                "the selected local model does not match the requested offline source".into(),
            );
        }
    }
    let mut requested = options.clone();
    requested.model_cache = Some(store.path().to_owned());
    requested.revision.get_or_insert(origin.revision.clone());
    if requested.filename.is_none() {
        requested.filename = origin.filename.clone();
    }
    if requested.tensor_parallel.is_empty() {
        requested.tensor_parallel = origin.sizes.clone();
    }
    let primary = store.primary(&entry)?;
    let resolved = crate::trust::resolve_identity_with(
        &primary,
        publisher_keys,
        supplied_profiles,
        &requested,
        trust_store,
    )?;
    verify_offline_binding(
        &entry,
        origin.expected_envelope,
        &origin.repository,
        &origin.revision,
        &resolved.profiles,
    )?;
    if options.offline {
        return Ok(entry);
    }
    let access_token = token(options)?;
    let staging = TransportStaging::new(store.path(), options.cache_dir.as_deref())?;
    let api = ApiBuilder::from_cache(Cache::new(staging.path().to_owned()))
        .with_endpoint(endpoint.into())
        .with_token(access_token.clone())
        .with_progress(false)
        .with_retries(3)
        .build()?;
    let revision = requested.revision.as_deref().unwrap();
    eprintln!("Checking model origin: {}@{revision}", origin.repository);
    let info = api.repo(Repo::with_revision(origin.repository.clone(), RepoType::Model, revision.into()))
        .info().map_err(|error| format!("cannot check {}@{revision} for updates: {error}; the local version is unchanged (use --offline to skip the check)", origin.repository))?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    if info.sha == origin.commit
        && revision == origin.revision
        && requested.filename == origin.filename
        && requested.tensor_parallel == origin.sizes
    {
        return Ok(entry);
    }
    eprintln!("Updating model from {}@{revision}", origin.repository);
    download_for_model_at(
        endpoint,
        &origin.repository,
        &requested,
        gpu_count,
        access_token,
        publisher_keys,
        trust_store,
        supplied_profiles,
        Some(entry.model()),
    )
}

fn authenticated_source_envelope(
    primary: &Path,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<[u8; 32]> {
    use flare_helios::package::{conversion, reader};
    use flare_helios::runtime_package::VerifiedPackage;
    if conversion::form(primary)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return Ok(VerifiedPackage::open(primary, publishers, None)?.envelope_digest());
    }
    let metadata = reader::Metadata::open(primary)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
        .map_err(|error| error.to_string())?;
    let selection = *selections
        .first()
        .ok_or("authenticated source has no selected deployment")?;
    // The signed envelope and selected deployment are sufficient to establish the
    // identity. Opening payload companions here would require materializing the
    // Hub snapshot and duplicate the acquisition's closure work.
    Ok(flare_helios::package::protection::verify(
        &metadata.envelope,
        publishers,
        selection,
    )?
    .envelope_digest())
}

pub(crate) fn remember_download(
    root: &Path,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    options: &HubOptions,
    gpu_count: usize,
    entry: &flare_helios::package::cache::Entry,
    expected_envelope: [u8; 32],
) -> Result<()> {
    let store = flare_helios::package::cache::Root::open(root)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let mut filenames = vec![Some(filename)];
    if options.filename.is_none() {
        filenames.push(None);
    }
    for revision in [revision, commit] {
        for filename in &filenames {
            // The exact selection retains earlier TP entries; the unqualified
            // selector also permits subsets of the most recently acquired union.
            for selection in [sizes.as_slice(), &[]] {
                store.publish_source_ref(
                    &source_selector(repository, revision, *filename, selection),
                    commit,
                    entry,
                    expected_envelope,
                )?;
            }
        }
    }
    store.publish_entry_source(
        &source_selector(repository, revision, Some(filename), &sizes),
        commit,
        entry,
        expected_envelope,
    )?;
    Ok(())
}

#[cfg(test)]
fn offline_entry(
    root: &Path,
    repository: &str,
    revision: &str,
    options: &HubOptions,
    gpu_count: usize,
) -> Result<flare_helios::package::cache::Entry> {
    offline_entry_with_expected(root, repository, revision, options, gpu_count)
        .map(|(entry, _)| entry)
}

fn unbound_source(repository: &str, revision: &str) -> String {
    format!(
        "cached source reference for {repository}@{revision} predates signed-envelope binding; refresh source online or load cached entry by explicit path"
    )
}

fn offline_entry_with_expected(
    root: &Path,
    repository: &str,
    revision: &str,
    options: &HubOptions,
    gpu_count: usize,
) -> Result<(flare_helios::package::cache::Entry, [u8; 32])> {
    let store = flare_helios::package::cache::Root::open(root)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let unqualified = source_selector(repository, revision, options.filename.as_deref(), &[]);
    // The unqualified record establishes the latest resolved commit. Its own
    // expected envelope is not used when an exact-TP record supplies the entry.
    let Some(latest) = store.read_source_ref(&unqualified)? else {
        return Err(missing_source(repository, revision, options, &sizes, root).into());
    };
    let commit = latest.revision().to_owned();
    let exact = source_selector(repository, revision, options.filename.as_deref(), &sizes);
    // An exact TP entry is usable only at the last known revision. Preferring it
    // before the unqualified selector would let a moving branch fall back to an
    // older commit merely because that commit had the requested TP cached.
    if let Some((reference, entry)) = store.resolve_source_ref_record(&exact)? {
        if reference.revision() == commit {
            let Some(expected) = reference.expected_envelope() else {
                return Err(unbound_source(repository, revision).into());
            };
            return Ok((entry, expected));
        }
    }
    if let Some((reference, entry)) = store.resolve_source_ref_record(&unqualified)? {
        if reference.revision() == commit {
            let Some(expected) = reference.expected_envelope() else {
                return Err(unbound_source(repository, revision).into());
            };
            return Ok((entry, expected));
        }
    }
    Err(missing_source(repository, revision, options, &sizes, root).into())
}

fn verify_offline_binding(
    entry: &flare_helios::package::cache::Entry,
    expected: [u8; 32],
    repository: &str,
    revision: &str,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<()> {
    use flare_helios::package::{PayloadId, conversion, reader};
    use flare_helios::runtime_package::VerifiedPackage;
    let primary = entry_primary(entry)?;
    let actual = if conversion::form(&primary)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        VerifiedPackage::open(&primary, publishers, None)?.envelope_digest()
    } else {
        let metadata = reader::Metadata::open(&primary)?;
        *PayloadId::of(&metadata.envelope).as_bytes()
    };
    if actual != expected {
        return Err(format!(
            "cached package for {repository}@{revision} does not match the signed envelope recorded for its source; refresh source online or load cached entry by explicit path"
        )
        .into());
    }
    Ok(())
}

fn missing_source(
    repository: &str,
    revision: &str,
    options: &HubOptions,
    sizes: &[u32],
    root: &Path,
) -> String {
    format!(
        "package {repository}@{revision} (filename: {}, TP: {sizes:?}) is not available in the Flare cache {}; download it online first",
        options.filename.as_deref().unwrap_or("automatic"),
        root.display(),
    )
}

fn enforce_legacy_tensor_parallel(
    verified: &flare_helios::runtime_package::VerifiedPackage,
    sizes: &[u32],
) -> Result<()> {
    let ranks = verified
        .identity()
        .map_err(|error| error.to_string())?
        .tensor_parallel;
    if sizes.len() != 1 || sizes[0] != ranks {
        return Err(format!(
            "legacy package provides exactly tensor-parallel size {ranks}, requested {sizes:?}"
        )
        .into());
    }
    Ok(())
}

fn verify_offline_entry(
    entry: &flare_helios::package::cache::Entry,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<()> {
    use flare_helios::{
        package::{conversion, reader},
        runtime_package::VerifiedPackage,
    };
    let primary = entry_primary(entry)?;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    if conversion::form(&primary)?.is_none() {
        let verified = VerifiedPackage::open(&primary, publishers, None)?;
        // A node-side download with neither an explicit TP request nor a GPU
        // count does not constrain a legacy package's signed rank topology.
        if !options.tensor_parallel.is_empty() || gpu_count > 0 {
            enforce_legacy_tensor_parallel(&verified, &sizes)?;
        }
        verified.verify_stored_data()?;
    } else {
        let metadata = reader::Metadata::open(&primary)?;
        let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
            .map_err(|error| error.to_string())?;
        for selection in selections {
            VerifiedPackage::open_selected(&primary, publishers, selection)?
                .verify_stored_data()?;
        }
    }
    Ok(())
}

fn entry_primary(entry: &flare_helios::package::cache::Entry) -> Result<PathBuf> {
    let files = std::fs::read_dir(entry.directory())?
        .map(|file| Ok(file?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>>>()?;
    Ok(entry.directory().join(choose_file(files, None)?))
}

/// Verify stored signatures and compressed executable bytes, select signed deployments,
/// and install the ready union as one leased cache entry. No terms acceptance and no
/// executable decoding happen here; acceptance stays at load time.
/// Resolve a transport package into a local tree of real files.
///
/// Hub snapshots store each file as a symlink into the shared blob cache. The
/// package reader opens components with `O_NOFOLLOW`, so acquisition materializes
/// the signed location closure into acquisition staging: canonical targets are
/// hardlinked (or copied across devices) under the same relative paths, and the
/// tree is removed after installation. Nothing here is a persistent store.
fn resolve_transport(
    primary: &Path,
    names: &[String],
    model_root: &Path,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let directory = primary.parent().unwrap_or(Path::new("."));
    let mut files = vec![
        primary
            .file_name()
            .ok_or("package primary has no file name")?
            .to_string_lossy()
            .into_owned(),
    ];
    files.extend(names.iter().cloned());
    files.sort();
    files.dedup();
    let symlinked = files.iter().any(|name| {
        std::fs::symlink_metadata(directory.join(name))
            .map(|info| info.file_type().is_symlink())
            .unwrap_or(false)
    });
    if !symlinked {
        return Ok((primary.to_owned(), None));
    }
    let staging = model_root.join("staging/acquisition");
    std::fs::create_dir_all(&staging)?;
    // One directory per acquisition, created exclusively so concurrent imports
    // never share a tree and cleanup only ever removes its own.
    let mut unique = 0u64;
    let tree = loop {
        let candidate = staging.join(format!("{}-{}", std::process::id(), unique));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                unique += 1;
                if unique > 1 << 16 {
                    return Err("cannot allocate acquisition staging".into());
                }
            }
            Err(error) => return Err(format!("cannot create acquisition staging: {error}").into()),
        }
    };
    let result = (|| {
        for name in &files {
            let source = directory.join(name);
            let canonical = std::fs::canonicalize(&source)
                .map_err(|error| format!("package file is unavailable: {name}: {error}"))?;
            let target = tree.join(name);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            match std::fs::hard_link(&canonical, &target) {
                Ok(()) => {}
                Err(_) => {
                    std::fs::copy(&canonical, &target)?;
                }
            }
        }
        let name = primary
            .strip_prefix(directory)
            .map_err(|_| "package primary escapes its directory")?;
        Ok(tree.join(name))
    })();
    match result {
        Ok(resolved) => Ok((resolved, Some(tree))),
        Err(error) => {
            let _ = std::fs::remove_dir_all(&tree);
            Err(error)
        }
    }
}

fn legacy_key(repository: &str) -> Result<String> {
    use flare_helios::package::cache;
    Ok(cache::model_key(repository)?)
}

enum PayloadAcquisition<'a> {
    Ready(flare_helios::package::cache::Entry),
    Planned {
        plan: flare_helios::package::cache::AcquisitionPlan<'a>,
        staging: flare_helios::package::cache::Staging,
    },
}

fn acquisition_model_key(
    source: &flare_helios::package::cache::PackageSource<'_>,
    repository: &str,
) -> Result<String> {
    use flare_helios::package::cache;
    match source
        .verified()
        .first()
        .map(|verified| verified.package().model_identity.clone())
    {
        Some(identity) if !identity.is_empty() => {
            Ok(cache::model_key(&identity).map_err(|error| error.to_string())?)
        }
        _ => Ok(cache::model_key(repository).map_err(|error| error.to_string())?),
    }
}

fn signed_hardware(
    source: &flare_helios::package::cache::PackageSource<'_>,
) -> Result<(Option<String>, Vec<u32>)> {
    use flare_helios::package::naming;
    let mut sizes = std::collections::BTreeSet::new();
    let mut package_platform: Option<&str> = None;
    let mut conventional_platform: Option<&str> = None;
    let mut conventional = true;
    for verified in source.verified() {
        let deployment = verified
            .deployment()
            .ok_or("selected deployment lacks an authenticated deployment body")?;
        if deployment.tensor_parallel == 0 {
            return Err("signed deployment has a zero tensor-parallel size".into());
        }
        sizes.insert(deployment.tensor_parallel);

        let signed_platform = verified.package().hardware_identity.as_str();
        if naming::PLATFORMS.contains(&signed_platform) {
            if let Some(previous) = package_platform {
                if previous != signed_platform {
                    return Err(format!(
                        "selected packages span {previous} and {signed_platform}; names require one device identity"
                    )
                    .into());
                }
            } else {
                package_platform = Some(signed_platform);
            }
        }

        let parsed = deployment
            .deployment_id
            .rsplit_once("-tp")
            .and_then(|(device, suffix)| {
                if !naming::PLATFORMS.contains(&device) {
                    return None;
                }
                let tensor_parallel = suffix.parse::<u32>().ok()?;
                (tensor_parallel == deployment.tensor_parallel).then_some(device)
            });
        match parsed {
            Some(device) => {
                if let Some(previous) = conventional_platform {
                    if previous != device {
                        conventional = false;
                    }
                } else {
                    conventional_platform = Some(device);
                }
            }
            None => conventional = false,
        }
    }
    if sizes.is_empty() {
        return Err("selected package lacks an authenticated tensor-parallel size".into());
    }
    let platform =
        package_platform.or_else(|| conventional.then_some(conventional_platform).flatten());
    Ok((platform.map(str::to_owned), sizes.into_iter().collect()))
}

fn acquisition_candidate(platform: Option<&str>, sizes: &[u32]) -> Result<String> {
    flare_helios::package::naming::deployment_candidate(
        platform,
        sizes,
        std::time::SystemTime::now(),
    )
    .ok_or_else(|| "acquisition naming requires at least one non-zero TP size".into())
}

/// Build the shared payload acquisition plan before any dependency fetch.
///
/// The returned plan owns cached object leases and the minimal remote file set.
/// An already-ready immutable entry with a matching remembered source avoids
/// planning entirely. A new entry reserves its readable device/TP/time name
/// before remote transfer and keeps that reservation through install.
#[allow(clippy::too_many_arguments)]
fn plan_payload_acquisition<'a>(
    primary: &Path,
    options: &HubOptions,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
    model_root: &Path,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<PayloadAcquisition<'a>> {
    use flare_helios::package::{cache, derivation, reader};

    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    // Resolve the primary through transport only. Companions are fetched by the
    // plan's remote set, never by a location-name closure.
    let (resolved, cleanup) = resolve_transport(primary, &[], model_root)?;
    let result = (|| {
        let metadata = reader::Metadata::open(&resolved)?;
        let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
            .map_err(|error| error.to_string())?;
        let source = cache::PackageSource::open(&resolved, publishers, selections[0])?
            .union(publishers, &selections)?;
        let model_key = acquisition_model_key(&source, repository)?;
        let store = cache::Root::open(model_root)?;

        // Warm reuse is keyed by the private source selector, then checked
        // against the signed primary and selected deployments before the hit.
        let mut selectors = Vec::new();
        for source_revision in [revision, commit] {
            for source_filename in [Some(filename), None] {
                for selection in [sizes.as_slice(), &[]] {
                    selectors.push(source_selector(
                        repository,
                        source_revision,
                        source_filename,
                        selection,
                    ));
                }
            }
        }
        for selector in selectors {
            let Ok(Some((stored_revision, entry))) = store.resolve_source_ref(&selector) else {
                continue;
            };
            if stored_revision == commit && store.verify_ready(&entry, &source).is_ok() {
                return Ok(PayloadAcquisition::Ready(entry));
            }
        }

        let (platform, signed_sizes) = signed_hardware(&source)?;
        let candidate = acquisition_candidate(platform.as_deref(), &signed_sizes)?;
        let staging = store.stage_named(&model_key, &candidate)?;
        let plan = store.plan_acquisition_with(source, &derivation::Limits::default(), progress)?;
        Ok(PayloadAcquisition::Planned { plan, staging })
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result
}

fn finish_payload_acquisition(
    primary: &Path,
    names: &[String],
    mut plan: flare_helios::package::cache::AcquisitionPlan<'_>,
    staging: flare_helios::package::cache::Staging,
    model_root: &Path,
) -> Result<flare_helios::package::cache::Entry> {
    use flare_helios::package::cache;
    let (resolved, cleanup) = resolve_transport(primary, names, model_root)?;
    let result = (|| {
        plan.relocate_source(&resolved)?;
        cache::Root::open(model_root)?.install_planned_named(plan, staging)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result.map_err(|error| error.to_string().into())
}

fn file_digest(path: &Path) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut context = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }
    Ok(context.finalize().into())
}

fn legacy_warm_entry(
    primary: &Path,
    options: &HubOptions,
    repository: &str,
    revision: &str,
    commit: &str,
    filename: &str,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
    model_root: &Path,
) -> Result<Option<flare_helios::package::cache::Entry>> {
    use flare_helios::package::cache;
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let (resolved, cleanup) = resolve_transport(primary, &[], model_root)?;
    let require_tp = !options.tensor_parallel.is_empty() || gpu_count > 0;
    let result = (|| {
        let store = cache::Root::open(model_root)?;
        let model_key = legacy_key(repository)?;
        let digest = file_digest(&resolved)?;

        // Preserve existing package-hash legacy entries without fetching companions,
        // but apply the same signed closure and TP checks as named source refs.
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::from_bytes(digest)
        );
        if let Ok(entry) = store.entry(&model_key, &old_id) {
            if let Ok(entry_primary) = entry_primary(&entry) {
                if file_digest(&entry_primary).is_ok_and(|candidate| candidate == digest) {
                    if let Ok(entry_verified) = flare_helios::runtime_package::VerifiedPackage::open(
                        &entry_primary,
                        publishers,
                        None,
                    ) {
                        let tp_ok = !require_tp
                            || entry_verified
                                .identity()
                                .is_ok_and(|identity| sizes == [identity.tensor_parallel]);
                        if tp_ok && entry_verified.verify_stored_data().is_ok() {
                            return Ok(Some(entry));
                        }
                    }
                }
            }
        }

        let mut selectors = Vec::new();
        for source_revision in [revision, commit] {
            for source_filename in [Some(filename), None] {
                for selection in [sizes.as_slice(), &[]] {
                    selectors.push(source_selector(
                        repository,
                        source_revision,
                        source_filename,
                        selection,
                    ));
                }
            }
        }
        for selector in selectors {
            let Ok(Some((stored_revision, entry))) = store.resolve_source_ref(&selector) else {
                continue;
            };
            if stored_revision != commit {
                continue;
            }
            let Ok(entry_primary) = entry_primary(&entry) else {
                continue;
            };
            if !file_digest(&entry_primary).is_ok_and(|candidate| candidate == digest) {
                continue;
            }
            let Ok(entry_verified) = flare_helios::runtime_package::VerifiedPackage::open(
                &entry_primary,
                publishers,
                None,
            ) else {
                continue;
            };
            let tp_ok = !require_tp
                || entry_verified
                    .identity()
                    .is_ok_and(|identity| sizes == [identity.tensor_parallel]);
            if tp_ok && entry_verified.verify_stored_data().is_ok() {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    result
}

fn install_ready(
    repository: &str,
    path: &Path,
    options: &HubOptions,
    model_root: &Path,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<flare_helios::package::cache::Entry> {
    use flare_helios::package::{cache, conversion};

    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    if !conversion::form(path)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return match plan_payload_acquisition(
            path,
            options,
            repository,
            options.revision.as_deref().unwrap_or(""),
            "",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.safetensors"),
            gpu_count,
            publishers,
            model_root,
            None,
        )? {
            PayloadAcquisition::Ready(entry) => Ok(entry),
            PayloadAcquisition::Planned { plan, staging } => {
                let names = plan.remote_files();
                finish_payload_acquisition(path, &names, plan, staging, model_root)
            }
        };
    }

    let names = flare_helios::runtime_package::package_companions(path)?;
    let (resolved, cleanup) = resolve_transport(path, &names, model_root)?;
    let install = (|| {
        let verified =
            flare_helios::runtime_package::VerifiedPackage::open(&resolved, publishers, None)?;
        if !options.tensor_parallel.is_empty() || gpu_count > 0 {
            enforce_legacy_tensor_parallel(&verified, &sizes)?;
        }
        let signed = verified.identity().map_err(|error| error.to_string())?;
        let platform =
            if flare_helios::package::naming::PLATFORMS.contains(&signed.hardware.as_str()) {
                Some(signed.hardware.as_str())
            } else {
                None
            };
        let candidate = acquisition_candidate(platform, &[signed.tensor_parallel])?;
        let store = cache::Root::open(model_root).map_err(|error| error.to_string())?;
        let model_key = legacy_key(repository)?;
        // Existing legacy entries keep their historical package-hash names.
        let primary_bytes = std::fs::read(&resolved).map_err(|error| error.to_string())?;
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::of(&primary_bytes)
        );
        if let Ok(entry) = store.entry(&model_key, &old_id) {
            return Ok(entry);
        }
        let staging = store
            .stage_named(&model_key, &candidate)
            .map_err(|error| error.to_string())?;
        let entry = store
            .import_legacy_named(&resolved, staging)
            .map_err(|error| error.to_string())?;
        Ok(entry)
    })();
    if let Some(tree) = cleanup {
        let _ = std::fs::remove_dir_all(tree);
    }
    install
}

/// The access token named by `--token-file`, for private repositories.
pub(crate) fn token(options: &HubOptions) -> Result<Option<String>> {
    options
        .token_file
        .as_ref()
        .map(|path| {
            let value = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read token file {}: {error}", path.display()))?;
            let token = value.trim();
            if token.is_empty() {
                return Err("Hugging Face token file is empty".into());
            }
            Ok(token.to_owned())
        })
        .transpose()
}

#[cfg(test)]
fn fetch_primary(
    api: &Api,
    cache: &Cache,
    repo: &Repo,
    requested: Option<&str>,
) -> Result<(PathBuf, Repo)> {
    fetch_primary_with_progress(api, cache, repo, requested, None)
}

fn fetch_primary_with_progress(
    api: &Api,
    cache: &Cache,
    repo: &Repo,
    requested: Option<&str>,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<(PathBuf, Repo)> {
    let info = api.repo(repo.clone()).info()?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    let filename = choose_file(
        info.siblings.into_iter().map(|file| file.rfilename),
        requested,
    )?;
    let pinned = Repo::with_revision(repo.url(), RepoType::Model, info.sha.clone());
    eprintln!("Resolved package: {}@{} / {filename}", repo.url(), info.sha);
    let path = fetch_file_with_progress(api, cache, &pinned, &filename, progress)?;
    cache.repo(repo.clone()).create_ref(&info.sha)?;
    Ok((path, pinned))
}

/// Fetch one file of an already resolved revision.
#[cfg(test)]
fn fetch_dependency(
    api: &Api,
    cache: &Cache,
    repo: &Repo,
    primary: &Path,
    name: &str,
) -> Result<()> {
    fetch_dependency_with_progress(api, cache, repo, primary, name, None)
}

fn fetch_dependency_with_progress(
    api: &Api,
    cache: &Cache,
    repo: &Repo,
    primary: &Path,
    name: &str,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<()> {
    let filename = primary
        .strip_prefix(snapshot(cache, repo, repo.revision()))
        .unwrap_or(primary);
    let base = Path::new(filename).parent().unwrap_or(Path::new(""));
    let relative = base.join(name).to_string_lossy().replace('\\', "/");
    fetch_file_with_progress(api, cache, repo, &relative, progress)?;
    Ok(())
}

/// Files the downloaded primary depends on.
///
/// Payload-format packages name their companions through signed location records,
/// so only the selected closure is fetched. Legacy packages keep their own
/// companion enumeration.
fn dependency_files(
    primary: &Path,
    options: &HubOptions,
    gpu_count: usize,
    publishers: &[flare_helios::runtime_package::Publisher],
) -> Result<Vec<String>> {
    use flare_helios::package::{derivation, protection, reader};

    let Ok(metadata) = reader::Metadata::open(primary) else {
        return Ok(flare_helios::runtime_package::package_companions(primary)?);
    };
    let sizes =
        crate::cache_support::requested_tensor_parallel(&options.tensor_parallel, gpu_count)
            .map_err(|error| error.to_string())?;
    let selections = crate::cache_support::select_deployments(&metadata, publishers, &sizes)
        .map_err(|error| error.to_string())?;
    let placed: std::collections::BTreeMap<Vec<u8>, String> = metadata
        .locations
        .locations
        .iter()
        .map(|location| (location.payload_id.clone(), location.file.clone()))
        .collect();
    // The shared cache closure decides which payload identities this selection
    // needs: its required payloads plus the derivation sources they depend on,
    // authorized only by the selected deployments' catalogs.
    let mut selected = Vec::with_capacity(selections.len());
    for selection in &selections {
        selected.push(
            protection::verify(&metadata.envelope, publishers, *selection)
                .map_err(|error| error.to_string())?,
        );
    }
    let needed = flare_helios::package::cache::selected_derivation_closure(
        &metadata,
        &selected,
        &derivation::Limits::default(),
    )
    .map_err(|error| error.to_string())?;
    let mut files: Vec<String> = needed
        .iter()
        .filter_map(|id| placed.get(id))
        .filter(|file| !file.is_empty())
        .cloned()
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// Reuse the repository's content-addressed blob even when this revision has no pointer yet.
#[cfg(test)]
fn fetch_file(api: &Api, cache: &Cache, pinned: &Repo, filename: &str) -> Result<PathBuf> {
    fetch_file_with_progress(api, cache, pinned, filename, None)
}

struct NetworkProgress {
    progress: flare_helios::package::progress::AcquisitionProgress,
    skip_next: bool,
}

impl hf_hub::api::Progress for NetworkProgress {
    fn init(&mut self, _size: usize, _filename: &str) {
        // The downloader reports the already-present prefix with update()
        // immediately after init. That is not network traffic for this attempt.
        self.skip_next = true;
    }

    fn update(&mut self, size: usize) {
        if self.skip_next {
            self.skip_next = false;
            return;
        }
        self.progress.record_network_bytes(size as u64);
    }

    fn finish(&mut self) {}
}

fn fetch_file_with_progress(
    api: &Api,
    cache: &Cache,
    pinned: &Repo,
    filename: &str,
    progress: Option<&flare_helios::package::progress::AcquisitionProgress>,
) -> Result<PathBuf> {
    let cached_repo = cache.repo(pinned.clone());
    let path = cached_repo.pointer_path(pinned.revision()).join(filename);
    if path.is_file() {
        return Ok(path);
    }
    let remote = api.repo(pinned.clone());
    let metadata = api.metadata(&remote.url(filename))?;
    let etag = metadata.etag();
    if metadata.commit_hash() != pinned.revision()
        || etag.is_empty()
        || !etag
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err("Hugging Face returned an invalid file identity".into());
    }
    let blob = cached_repo.blob_path(etag);
    std::fs::create_dir_all(blob.parent().unwrap())?;
    // hf-hub locks its transfer but replaces an existing blob. Serialize our lookup
    // and transfer together so concurrent Flare downloads preserve shared inodes.
    let _lock = blob_lock(&blob)?;
    if path.is_file() {
        return Ok(path);
    }
    match std::fs::symlink_metadata(&blob) {
        Ok(info) => {
            if !info.is_file() || info.len() != metadata.size() as u64 {
                return Err(format!("invalid cached Hugging Face blob: {}", blob.display()).into());
            }
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::hard_link(&blob, &path).map_err(|error| {
                format!(
                    "cannot hardlink cached package file {}: {error}",
                    path.display()
                )
            })?;
            eprintln!("Reusing cached model file: {filename}");
            Ok(path)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Downloading model file: {filename}...");
            let downloaded = match progress {
                Some(progress) => remote.download_with_progress(
                    filename,
                    NetworkProgress {
                        progress: progress.clone(),
                        skip_next: true,
                    },
                )?,
                None => remote.download(filename)?,
            };
            // Without symlink privileges, hf-hub moves the blob into the snapshot.
            // Retain a cache link so other revisions can still reuse those bytes.
            #[cfg(windows)]
            if !blob.exists() {
                std::fs::hard_link(&downloaded, &blob)?;
            }
            Ok(downloaded)
        }
        Err(error) => Err(error.into()),
    }
}

fn blob_lock(blob: &Path) -> Result<std::fs::File> {
    let path = blob.with_extension("flare.lock");
    #[cfg(unix)]
    let lock = {
        use std::os::unix::{fs::OpenOptionsExt, io::AsRawFd};
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        file
    };
    #[cfg(windows)]
    let lock = {
        let file = crate::windows::private_file(&path, false)?;
        crate::windows::lock(&file)?;
        file
    };
    Ok(lock)
}

fn snapshot(cache: &Cache, repo: &Repo, commit: &str) -> PathBuf {
    cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(commit)
}

/// Cached primary inside the Hub snapshot, without any second export tree.
/// Payload companions are not candidates for the package primary.
fn is_canonical_primary(file: &str) -> bool {
    let segments: Vec<&str> = file.split('/').collect();
    segments.len() == 1 || (segments.len() == 2 && segments[1] == "model.safetensors")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    const COMMIT: &str = "0123456789012345678901234567890123456789";

    /// Build a minimal signed payload package plus its trusted public profile.
    ///
    /// Used by Orbit's offline success fixture so the node verifies a real
    /// signature and a real payload closure without HOME or embedded profiles.
    pub(crate) fn signed_payload_fixture(
        parent: &std::path::Path,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        flare_helios::runtime_package::Publisher,
        String,
    ) {
        signed_payload_fixture_with(parent, "gb10-tp1", "sm120")
    }

    fn signed_payload_fixture_with(
        parent: &std::path::Path,
        deployment_id: &str,
        hardware_identity: &str,
    ) -> (
        std::path::PathBuf,
        std::path::PathBuf,
        flare_helios::runtime_package::Publisher,
        String,
    ) {
        signed_payload_fixture_identity(parent, deployment_id, hardware_identity, "owner/offline")
    }

    fn signed_payload_fixture_identity(
        parent: &Path, deployment_id: &str, hardware_identity: &str, model: &str,
    ) -> (PathBuf, PathBuf, flare_helios::runtime_package::Publisher, String) {
        use flare_helios::package::{PayloadId, container, export, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let package = parent.join("package");
        std::fs::create_dir_all(&package).unwrap();
        let weights = b"offline orbit weights".repeat(4);
        let id = PayloadId::of(&weights);
        let raw = parent.join("weights.bin");
        std::fs::write(&raw, &weights).unwrap();
        let graph = parent.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Offline Orbit".into(),
            namespace: "local-development".into(),
            public_key,
            development: true,
        };
        let contract = flare_helios::runtime_package::wire::ServingContract {
            model_id: "orbit-offline".into(),
            identity: "orbit-offline-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: deployment_id.into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weights.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: weights.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: model.into(),
                hardware_identity: hardware_identity.into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "local-development".into(),
                    official_source: "local-development".into(),
                    development: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: deployment_id.into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(contract),
                    ..Default::default()
                },
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            weights.len() as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: 11,
                    }],
                }],
            }],
            package_id: [23u8; 32],
            public_key,
            accepted: "local-development".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&package)).unwrap();
        let profile = parent.join("publisher.json");
        std::fs::write(
            &profile,
            serde_json::to_vec(&vec![publisher.clone()]).unwrap(),
        )
        .unwrap();
        (package, profile, publisher, COMMIT.to_owned())
    }

    pub(crate) fn fixture(companions: &[&str]) -> Vec<u8> {
        fn bytes(field: u8, body: &[u8]) -> Vec<u8> {
            let mut out = vec![field * 8 + 2];
            let mut n = body.len();
            while n >= 128 {
                out.push((n as u8 & 127) | 128);
                n >>= 7;
            }
            out.push(n as u8);
            out.extend_from_slice(body);
            out
        }
        // ProtectedIndex -> PublisherMetadata -> ModelIndex, with unsigned discovery metadata.
        let index = [8, if companions.is_empty() { 4 } else { 5 }];
        let metadata = bytes(1, &index);
        let mut envelope = vec![8, 4];
        for (kind, body) in [(1, metadata), (2, vec![]), (3, vec![])] {
            envelope.extend(bytes(4, &[vec![8, kind], bytes(2, &body)].concat()));
        }
        let mut data = Vec::new();
        for name in companions {
            data.extend(bytes(4, &bytes(3, name.as_bytes())));
        }
        let rank = [bytes(2, &[]), bytes(5, &data)].concat();
        envelope.extend(bytes(5, &[vec![8, 4], bytes(2, &rank)].concat()));
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
        });
        let mut result = 4088u64.to_le_bytes().to_vec();
        result.extend(serde_json::to_vec(&header).unwrap());
        result.resize(4096, b' ');
        result.extend(envelope);
        result
    }

    fn origin_fixture(
        parent: &Path,
    ) -> (
        flare_helios::package::cache::Root,
        flare_helios::package::cache::Entry,
        flare_helios::runtime_package::Publisher,
        HubOptions,
    ) {
        let (package, _, publisher, _) = signed_payload_fixture(parent);
        let options = HubOptions {
            model_cache: Some(parent.join("models")),
            ..Default::default()
        };
        let store = flare_helios::package::cache::Root::open(options.model_cache.as_ref().unwrap())
            .unwrap();
        let entry = install_ready(
            "publisher/prepared",
            &package,
            &options,
            store.path(),
            1,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        let digest =
            authenticated_source_envelope(&package, &options, 1, std::slice::from_ref(&publisher))
                .unwrap();
        remember_download(
            store.path(),
            "publisher/prepared",
            "gb10",
            COMMIT,
            "model.safetensors",
            &options,
            1,
            &entry,
            digest,
        )
        .unwrap();
        store.publish_entry(&entry).unwrap();
        (store, entry, publisher, options)
    }

    #[test]
    fn model_refresh_checks_recorded_repository_without_downloading_unchanged_payloads() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, options) = origin_fixture(temp.path());
        assert_ne!(
            entry.model(),
            flare_helios::package::cache::model_key("publisher/prepared").unwrap()
        );
        let original = entry.id().to_owned();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/publisher/prepared/revision/gb10".into(),
            metadata_response_with(&["model.safetensors"]),
        )]);
        let selected = refresh_model_at(
            &endpoint,
            &store,
            entry,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher],
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(selected.id(), original);
        assert!(store.evict(selected.model(), selected.id()).is_err());
    }

    #[test]
    fn model_refresh_pinned_and_offline_never_contact_origin() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, mut options) = origin_fixture(temp.path());
        let model = entry.model().to_owned();
        let version = entry.id().to_owned();
        options.token_file = Some(temp.path().join("missing-token"));
        for (pinned, offline) in [(true, false), (false, true)] {
            options.offline = offline;
            let selected = refresh_model_at(
                "http://127.0.0.1:0",
                &store,
                store.entry(&model, &version).unwrap(),
                pinned,
                &options,
                1,
                &[],
                None,
                std::slice::from_ref(&publisher),
            )
            .unwrap();
            assert_eq!(selected.id(), version);
        }
        options.offline = false;
        options.revision = Some("different".into());
        assert!(
            refresh_model_at(
                "http://127.0.0.1:0",
                &store,
                entry,
                true,
                &options,
                1,
                &[],
                None,
                &[publisher]
            )
            .err()
            .unwrap()
            .to_string()
            .contains("pinned")
        );
    }

    #[test]
    fn model_refresh_installs_update_and_preserves_previous_version_origin() {
        let temp = tempfile::tempdir().unwrap();
        let (store, old, publisher, options) = origin_fixture(&temp.path().join("original"));
        let old_id = old.id().to_owned();
        // The selected entry was acquired before the commit now served by the branch.
        let old_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let origin = entry_origin(&store, &old).unwrap().unwrap();
        store
            .publish_entry_source(
                &source_selector(
                    "publisher/prepared",
                    "gb10",
                    Some("model.safetensors"),
                    &[1],
                ),
                old_commit,
                &old,
                origin.expected_envelope,
            )
            .unwrap();
        let (package, _, updated_publisher, _) =
            signed_payload_fixture(&temp.path().join("updated"));
        let mut routes = package_routes(&package, "publisher/prepared", "gb10");
        let response = routes
            .get_mut("/api/models/publisher/prepared/revision/gb10")
            .unwrap();
        response.1 = response.0.clone();
        let (endpoint, worker, _) = mock_hub_counted(routes);
        let selected = refresh_model_at(
            &endpoint,
            &store,
            old,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher, updated_publisher],
        )
        .unwrap();
        worker.join().unwrap();
        assert_ne!(selected.id(), old_id);
        assert_eq!(
            store.resolve(selected.model()).unwrap().unwrap().id(),
            selected.id()
        );
        assert_eq!(
            entry_origin(&store, &selected).unwrap().unwrap().commit,
            COMMIT
        );
        let previous = store.entry(selected.model(), &old_id).unwrap();
        assert_eq!(
            entry_origin(&store, &previous).unwrap().unwrap().commit,
            old_commit
        );
    }

    #[test]
    fn model_refresh_rejects_changed_identity_and_untrusted_update_without_selecting_it() {
        for (model, trusted, expected_error) in [
            ("owner/other", true, "different model"),
            ("owner/offline", false, "publisher"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let (store, old, publisher, options) = origin_fixture(&temp.path().join("original"));
            let key = old.model().to_owned();
            let id = old.id().to_owned();
            let origin = entry_origin(&store, &old).unwrap().unwrap();
            store
                .publish_entry_source(
                    &source_selector(
                        "publisher/prepared",
                        "gb10",
                        Some("model.safetensors"),
                        &[1],
                    ),
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    &old,
                    origin.expected_envelope,
                )
                .unwrap();
            let (package, _, other, _) = signed_payload_fixture_identity(
                &temp.path().join("updated"),
                "gb10-tp1",
                "sm120",
                model,
            );
            let mut routes = package_routes(&package, "publisher/prepared", "gb10");
            let response = routes
                .get_mut("/api/models/publisher/prepared/revision/gb10")
                .unwrap();
            response.1 = response.0.clone();
            let (endpoint, worker, _) = mock_hub_counted(routes);
            let mut profiles = vec![publisher];
            if trusted {
                profiles.push(other);
            }
            let error = refresh_model_at(
                &endpoint,
                &store,
                old,
                false,
                &options,
                1,
                &[],
                None,
                &profiles,
            )
            .err()
            .unwrap()
            .to_string();
            worker.join().unwrap();
            assert!(error.contains(expected_error), "{error}");
            assert_eq!(store.resolve(&key).unwrap().unwrap().id(), id);
        }
    }

    #[test]
    fn model_origin_recovers_hub_refs_and_ignores_rank_transfer_refs() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, _, _) = origin_fixture(temp.path());
        for record in std::fs::read_dir(store.path().join("origins")).unwrap() {
            std::fs::remove_file(record.unwrap().path()).unwrap();
        }
        store
            .publish_source_ref("distributed-export-rank0", "export", &entry, [1; 32])
            .unwrap();
        let origin = entry_origin(&store, &entry).unwrap().unwrap();
        assert_eq!(origin.repository, "publisher/prepared");
        assert_eq!(origin.revision, "gb10");
        assert_eq!(origin.sizes, vec![1]);
        for record in std::fs::read_dir(store.path().join("sources")).unwrap() {
            std::fs::remove_file(record.unwrap().path()).unwrap();
        }
        assert_eq!(entry_origin(&store, &entry).unwrap(), Some(origin));
    }

    #[test]
    fn model_refresh_failed_update_keeps_default() {
        let temp = tempfile::tempdir().unwrap();
        let (store, entry, publisher, options) = origin_fixture(temp.path());
        let model = entry.model().to_owned();
        let version = entry.id().to_owned();
        let response =
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/publisher/prepared/revision/gb10".into(),
            response,
        )]);
        let error = refresh_model_at(
            &endpoint,
            &store,
            entry,
            false,
            &options,
            1,
            &[],
            None,
            &[publisher],
        )
        .err()
        .unwrap()
        .to_string();
        worker.join().unwrap();
        assert!(error.contains("--offline"), "{error}");
        assert_eq!(store.resolve(&model).unwrap().unwrap().id(), version);
    }

    #[test]
    fn local_paths_and_repository_revisions_are_distinct() {
        assert!(is_repository(Path::new("owner/model")));
        assert!(is_repository(Path::new("owner/model@release/gb202")));
        for path in [
            "model.safetensors",
            "dir/model.safetensors",
            "./owner/model",
            "/owner/model",
            "../owner/model",
            "C:\\models\\model.safetensors",
        ] {
            assert!(!is_repository(Path::new(path)), "{path}");
        }
        assert_eq!(
            repository("owner/model@gb10").unwrap(),
            ("owner/model", Some("gb10"))
        );
        for source in [
            "owner",
            "owner/model/extra",
            "../model",
            "owner/model@",
            "owner/model@../../other",
        ] {
            assert!(repository(source).is_err(), "{source}");
        }
    }

    #[test]
    fn package_selection_rejects_ambiguous_and_escaping_filenames() {
        assert_eq!(
            choose_file(["README.md".into(), "model.safetensors".into()], None).unwrap(),
            "model.safetensors"
        );
        let error = choose_file(["tp4.safetensors".into(), "tp2.safetensors".into()], None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--filename"));
        assert!(error.contains("tp2.safetensors\n  tp4.safetensors"));
        for file in [
            "../model.safetensors",
            "/model.safetensors",
            "C:\\model.safetensors",
            "weights.json",
        ] {
            assert!(choose_file([], Some(file)).is_err(), "{file}");
        }
    }

    fn mock_hub(responses: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
        mock_hub_steps(responses, |_| {})
    }

    fn mock_hub_steps(
        responses: Vec<(String, Vec<u8>)>,
        mut after_response: impl FnMut(usize) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            for (step, (expected, response)) in responses.into_iter().enumerate() {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with(&format!("GET {expected} ")), "{line}");
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                socket.write_all(&response).unwrap();
                after_response(step);
            }
        });
        (endpoint, worker)
    }

    /// Serve a snapshot by path: first request for a file returns its metadata,
    /// later requests return its body. Unknown paths panic, so an unselected file
    /// request fails the test.
    fn mock_hub_snapshot(
        routes: std::collections::HashMap<String, (Vec<u8>, Vec<u8>)>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            let mut seen = std::collections::HashSet::new();
            let mut idle = 0;
            loop {
                let (mut socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        idle += 1;
                        if idle > 200 {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("accept failed: {error}"),
                };
                idle = 0;
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .expect("request path")
                    .to_owned();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                let (metadata, body) = routes.get(&path).unwrap_or_else(|| {
                    panic!("unexpected request {path}");
                });
                let response = if seen.insert(path.clone()) {
                    metadata
                } else {
                    body
                };
                socket.write_all(response).unwrap();
            }
        });
        (endpoint, worker)
    }

    #[derive(Default, Debug)]
    struct TransferLog {
        files: std::collections::BTreeMap<String, usize>,
        bytes: usize,
    }

    /// Serve a package snapshot and record body downloads by path and byte count.
    fn mock_hub_counted(
        routes: std::collections::HashMap<String, (Vec<u8>, Vec<u8>)>,
    ) -> (
        String,
        thread::JoinHandle<()>,
        std::sync::Arc<std::sync::Mutex<TransferLog>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let log = std::sync::Arc::new(std::sync::Mutex::new(TransferLog::default()));
        let recorder = std::sync::Arc::clone(&log);
        let worker = thread::spawn(move || {
            let mut seen = std::collections::HashSet::new();
            let mut idle = 0;
            let mut served = 0;
            loop {
                let (mut socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if served > 0 {
                            idle += 1;
                            if idle > 200 {
                                break;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("accept failed: {error}"),
                };
                idle = 0;
                served += 1;
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .expect("request path")
                    .to_owned();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                let (metadata, body) = routes.get(&path).unwrap_or_else(|| {
                    panic!("unexpected request {path}");
                });
                let first = seen.insert(path.clone());
                if !first {
                    let mut log = recorder.lock().unwrap();
                    *log.files.entry(path.clone()).or_insert(0) += 1;
                    log.bytes += body.len();
                }
                socket
                    .write_all(if first { metadata } else { body })
                    .unwrap();
            }
        });
        (endpoint, worker, log)
    }

    fn package_files(root: &Path) -> Vec<String> {
        fn walk(root: &Path, directory: &Path, files: &mut Vec<String>) {
            for entry in std::fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    walk(root, &path, files);
                } else {
                    files.push(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                    );
                }
            }
        }
        let mut files = Vec::new();
        walk(root, root, &mut files);
        files.sort();
        files
    }

    fn package_routes(
        root: &Path,
        repository: &str,
        revision: &str,
    ) -> std::collections::HashMap<String, (Vec<u8>, Vec<u8>)> {
        use sha2::{Digest, Sha256};
        let files = package_files(root);
        let names: Vec<&str> = files.iter().map(String::as_str).collect();
        let mut routes = std::collections::HashMap::new();
        routes.insert(
            format!("/api/models/{repository}/revision/{revision}"),
            (metadata_response_with(&names), Vec::new()),
        );
        for file in files {
            let body = std::fs::read(root.join(&file)).unwrap();
            let etag = format!("{:x}", Sha256::digest(&body));
            routes.insert(
                format!("/{repository}/resolve/{COMMIT}/{file}"),
                (
                    file_metadata(COMMIT, &etag, body.len()),
                    file_body(&etag, &body),
                ),
            );
        }
        routes
    }

    fn counted_download(
        package: &Path,
        repository: &str,
        revision: &str,
        options: &HubOptions,
        publisher: &flare_helios::runtime_package::Publisher,
    ) -> (TransferLog, String) {
        let routes = package_routes(package, repository, revision);
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let entry = download_with_token_at(
            &endpoint,
            &format!("{repository}@{revision}"),
            options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(publisher),
        )
        .unwrap();
        let entry_id = entry.id().to_owned();
        drop(entry);
        worker.join().unwrap();
        let log = log.lock().unwrap();
        (
            TransferLog {
                files: log.files.clone(),
                bytes: log.bytes,
            },
            entry_id,
        )
    }

    fn counted_download_entry(
        endpoint: &str,
        repository: &str,
        options: &HubOptions,
        publisher: &flare_helios::runtime_package::Publisher,
        log: &std::sync::Arc<std::sync::Mutex<TransferLog>>,
    ) -> (flare_helios::package::cache::Entry, TransferLog) {
        let entry = download_with_token_at(
            endpoint,
            &format!("{repository}@gb10"),
            options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(publisher),
        )
        .unwrap();
        let snapshot = {
            let log = log.lock().unwrap();
            TransferLog {
                files: log.files.clone(),
                bytes: log.bytes,
            }
        };
        (entry, snapshot)
    }

    fn rewrite_payload_primary(
        directory: &Path,
        locations: Vec<flare_helios::package::schema::wire::PayloadLocation>,
        derivations: Option<&flare_helios::package::schema::wire::DerivationSet>,
        tensors: &[(&str, &[u8])],
    ) {
        use flare_helios::package::{container, reader, schema::wire};
        use prost::Message;
        let metadata = reader::Metadata::open(directory).unwrap();
        let manifest = metadata.primary().tensor(reader::MANIFEST).unwrap();
        let replaced: std::collections::BTreeSet<Vec<u8>> = locations
            .iter()
            .map(|location| location.payload_id.clone())
            .collect();
        let mut preserved: Vec<wire::PayloadLocation> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !replaced.contains(&location.payload_id))
            .cloned()
            .collect();
        preserved.extend(locations);
        let locations = wire::LocationSet {
            version: metadata.locations.version,
            locations: preserved,
            available_deployments: metadata.locations.available_deployments.clone(),
        };
        let path = directory.join("model.safetensors");
        let temporary = directory.join("model.rebuilt");
        let mut output = std::fs::File::create(&temporary).unwrap();
        let mut writer = container::Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
        writer
            .add(
                reader::MANIFEST,
                manifest.len(),
                container::Source::Slice(manifest),
            )
            .unwrap();
        let encoded = locations.encode_to_vec();
        writer
            .add(
                reader::LOCATIONS,
                encoded.len() as u64,
                container::Source::Bytes(encoded),
            )
            .unwrap();
        if let Some(derivations) = derivations {
            let encoded = derivations.encode_to_vec();
            writer
                .add(
                    reader::DERIVATIONS,
                    encoded.len() as u64,
                    container::Source::Bytes(encoded),
                )
                .unwrap();
        }
        for name in metadata.primary().tensor_names() {
            if name == reader::MANIFEST
                || name == reader::LOCATIONS
                || name == reader::DERIVATIONS
                || name.starts_with(container::ALIGNMENT_PREFIX)
                || tensors.iter().any(|(replacement, _)| *replacement == name)
            {
                continue;
            }
            let tensor = metadata.primary().tensor(name).unwrap();
            writer
                .add(name, tensor.len(), container::Source::Slice(tensor))
                .unwrap();
        }
        for (name, bytes) in tensors {
            writer
                .add(
                    name,
                    bytes.len() as u64,
                    container::Source::Bytes(bytes.to_vec()),
                )
                .unwrap();
        }
        writer.finish().unwrap();
        output.sync_all().unwrap();
        std::fs::rename(temporary, path).unwrap();
    }

    fn location_bytes(
        directory: &Path,
        location: &flare_helios::package::schema::wire::PayloadLocation,
    ) -> Vec<u8> {
        use flare_helios::package::container;
        let container = if location.file.is_empty() {
            container::Container::open(&directory.join("model.safetensors")).unwrap()
        } else {
            container::Container::open(&directory.join(&location.file)).unwrap()
        };
        let slice = container.tensor(&location.tensor).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn blob_count(root: &Path) -> usize {
        std::fs::read_dir(root.join("blobs"))
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().is_file())
            .count()
    }

    #[test]
    fn warm_download_ignores_malformed_optional_hints_without_dependency_transfer() {
        for variant in ["malformed", "overlimit"] {
            let temp = tempfile::tempdir().unwrap();
            let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
            let metadata = flare_helios::package::reader::Metadata::open(&package).unwrap();
            let locations = metadata.locations.locations.clone();
            let hints = if variant == "malformed" {
                flare_helios::package::schema::wire::DerivationSet {
                    version: flare_helios::package::derivation::DERIVATION_VERSION + 1,
                    derivations: Vec::new(),
                }
            } else {
                let count =
                    flare_helios::package::derivation::Limits::default().max_derivations + 1;
                flare_helios::package::schema::wire::DerivationSet {
                    version: flare_helios::package::derivation::DERIVATION_VERSION,
                    derivations: (0..count)
                        .map(|_| flare_helios::package::schema::wire::Derivation {
                            target_payload_id: Vec::new(),
                            operations: Vec::new(),
                        })
                        .collect(),
                }
            };
            rewrite_payload_primary(&package, locations, Some(&hints), &[]);
            let model_root = temp.path().join("models");
            let options = HubOptions {
                revision: Some("gb10".into()),
                tensor_parallel: vec![1],
                model_cache: Some(model_root.clone()),
                ..Default::default()
            };
            let routes = package_routes(&package, "owner/offline", "gb10");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (cold_entry, cold) =
                counted_download_entry(&endpoint, "owner/offline", &options, &publisher, &log);
            worker.join().unwrap();
            let cold_id = cold_entry.id().to_owned();
            let blobs = blob_count(&model_root);
            drop(cold_entry);

            let routes = package_routes(&package, "owner/offline", "gb10");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (warm_entry, warm) =
                counted_download_entry(&endpoint, "owner/offline", &options, &publisher, &log);
            assert_eq!(warm_entry.id(), cold_id, "{variant}: warm entry changed");
            assert!(
                warm.files
                    .keys()
                    .all(|file| file.ends_with("/model.safetensors")),
                "{variant}: optional hints caused dependency transfer: {warm:?}"
            );
            assert_eq!(
                blob_count(&model_root),
                blobs,
                "{variant}: optional hints caused materialization"
            );
            let key = flare_helios::package::cache::model_key("owner/offline").unwrap();
            assert_eq!(
                flare_helios::package::cache::Root::open(&model_root)
                    .unwrap()
                    .ref_entry(&key)
                    .unwrap()
                    .as_deref(),
                Some(cold_id.as_str()),
                "{variant}: warm success changed the selected entry unexpectedly"
            );
            drop(warm_entry);
            worker.join().unwrap();
            let _ = cold;
        }
    }

    #[test]
    fn offline_resolution_rejects_corrupt_embedded_payload_before_default_update() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &options,
            &publishers[0],
            &log,
        );
        let selected = entry.id().to_owned();
        worker.join().unwrap();

        let locations = flare_helios::package::reader::Metadata::open(entry.directory())
            .unwrap()
            .locations
            .locations;
        rewrite_payload_primary(
            entry.directory(),
            locations,
            None,
            &[("frontend.tokenizer", b"corrupt embedded frontend")],
        );
        drop(entry);

        let offline = HubOptions {
            offline: true,
            ..options
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
        ) {
            Ok(_) => panic!("offline resolution accepted corrupt embedded frontend bytes"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("payload")
                || error.to_string().contains("digest")
                || error.to_string().contains("verification"),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        assert_eq!(
            flare_helios::package::cache::Root::open(&model_root)
                .unwrap()
                .ref_entry(&key)
                .unwrap()
                .as_deref(),
            Some(selected.as_str()),
            "offline verification failure changed the default"
        );
    }

    #[test]
    fn offline_source_ref_rejects_a_different_valid_envelope_without_mutation() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let (_other_scratch, other_primary, other_publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &options,
            &publishers[0],
            &log,
        );
        let selected = entry.id().to_owned();
        worker.join().unwrap();
        let ref_selector = source_selector("sel/model", "tp1", None, &[]);
        let expected = flare_helios::package::cache::Root::open(&model_root)
            .unwrap()
            .read_source_ref(&ref_selector)
            .unwrap()
            .unwrap()
            .expected_envelope()
            .unwrap();
        let installed_primary = entry.directory().join("model.safetensors");
        let mut permissions = std::fs::metadata(&installed_primary).unwrap().permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(&installed_primary, permissions).unwrap();
        std::fs::copy(&other_primary, &installed_primary).unwrap();
        drop(entry);

        let mut all_publishers = publishers.clone();
        all_publishers.extend(other_publishers);
        let offline = HubOptions {
            offline: true,
            ..options
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &all_publishers,
        ) {
            Ok(_) => panic!("offline resolution accepted a different signed envelope"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("does not match the signed envelope recorded for its source"),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(selected.as_str()),
            "envelope mismatch changed the default"
        );
        assert_eq!(
            store
                .read_source_ref(&ref_selector)
                .unwrap()
                .unwrap()
                .expected_envelope(),
            Some(expected),
            "envelope mismatch changed the source reference"
        );
    }

    fn replace_source_ref_with_legacy_record(
        root: &Path,
        repository: &str,
        revision: &str,
        entry: &flare_helios::package::cache::Entry,
    ) -> PathBuf {
        let selector = source_selector(repository, revision, None, &[]);
        let path = root
            .join("sources")
            .join(flare_helios::package::PayloadId::of(selector.as_bytes()).to_string());
        std::fs::write(
            &path,
            serde_json::to_vec(&(selector, revision, entry.model(), entry.id())).unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn legacy_unbound_source_ref_fails_offline_and_refreshes_online_without_dependencies() {
        let (_source_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (entry, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &options,
            &publishers[0],
            &log,
        );
        let selected = entry.id().to_owned();
        worker.join().unwrap();
        let ref_path = replace_source_ref_with_legacy_record(
            &model_root,
            "sel/model",
            "tp1",
            &entry,
        );
        drop(entry);

        let offline = HubOptions {
            offline: true,
            ..options.clone()
        };
        let error = match download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
        ) {
            Ok(_) => panic!("offline resolution accepted an unbound source reference"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(
                "refresh source online or load cached entry by explicit path"
            ),
            "unexpected offline failure: {error}"
        );
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(selected.as_str()),
            "unbound offline failure changed the default"
        );
        assert_eq!(
            store
                .read_source_ref(&source_selector("sel/model", "tp1", None, &[]))
                .unwrap()
                .unwrap()
                .expected_envelope(),
            None
        );
        assert!(ref_path.is_file());

        // Online resolution reuses the existing entry and writes a bound source
        // record without fetching dependency payloads.
        let routes = package_routes(source_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (refreshed, warm) = counted_download_entry(
            &endpoint,
            "sel/model",
            &options,
            &publishers[0],
            &log,
        );
        assert_eq!(refreshed.id(), selected);
        assert!(
            warm.files
                .keys()
                .all(|file| file.ends_with("/model.safetensors")),
            "source refresh fetched dependencies: {warm:?}"
        );
        worker.join().unwrap();
        drop(refreshed);
        let bound = store
            .read_source_ref(&source_selector("sel/model", "tp1", None, &[]))
            .unwrap()
            .unwrap();
        assert!(bound.expected_envelope().is_some());

        let selected_offline = download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &offline,
            1,
            None,
            &[],
            None,
            &publishers,
        )
        .expect("bound source reference must resolve offline");
        assert_eq!(selected_offline.id(), selected);
    }

    #[test]
    fn offline_binding_keeps_the_selected_ref_across_a_concurrent_update() {
        let (_first_scratch, first_primary, first_publishers) = selected_tp_package();
        let (_second_scratch, second_primary, second_publishers) = selected_tp_package();
        let first_root = first_primary.parent().unwrap();
        let second_root = second_primary.parent().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let first_options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let second_options = HubOptions {
            revision: Some("tp1".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let routes = package_routes(first_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (first, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &first_options,
            &first_publishers[0],
            &log,
        );
        let first_id = first.id().to_owned();
        worker.join().unwrap();

        // Capture the exact TP1 selection and its expected identity under its lease
        // before a concurrent acquisition updates the unqualified selector.
        let (leased, expected) =
            offline_entry_with_expected(&model_root, "sel/model", "tp1", &first_options, 1)
                .unwrap();
        assert_eq!(leased.id(), first_id);
        drop(first);

        let routes = package_routes(second_root, "sel/model", "tp1");
        let (endpoint, worker, log) = mock_hub_counted(routes);
        let (second, _) = counted_download_entry(
            &endpoint,
            "sel/model",
            &second_options,
            &second_publishers[0],
            &log,
        );
        let second_id = second.id().to_owned();
        assert_ne!(first_id, second_id);
        worker.join().unwrap();
        drop(second);

        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let unqualified = source_selector("sel/model", "tp1", None, &[]);
        assert_eq!(
            store
                .read_source_ref(&unqualified)
                .unwrap()
                .unwrap()
                .entry(),
            second_id
        );
        // The captured identity belongs to the exact TP1 entry, not the now-current
        // unqualified TP2 record.
        verify_offline_binding(&leased, expected, "sel/model", "tp1", &first_publishers).unwrap();

        let selected = download_with_token_at(
            "http://unused",
            "sel/model@gb10",
            &HubOptions {
                offline: true,
                ..first_options
            },
            1,
            None,
            &[],
            None,
            &first_publishers,
        )
        .expect("exact TP1 source binding must survive a concurrent unqualified update");
        assert_eq!(selected.id(), first_id);
        let key = flare_helios::package::cache::model_key("sel").unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(first_id.as_str())
        );
    }


    #[test]
    fn warm_download_repairs_corrupt_entry_from_signed_source() {
        let (_scratch, primary, publishers) = selected_tp_package();
        let source_root = primary.parent().unwrap();
        let source_metadata = flare_helios::package::reader::Metadata::open(&primary).unwrap();
        let source_envelope = source_metadata.envelope.clone();
        let source_primary = std::fs::read(&primary).unwrap();
        let other = tempfile::tempdir().unwrap();
        let (other_package, _, _, _) = signed_payload_fixture(other.path());
        let other_primary = other_package.join("model.safetensors");
        let key = flare_helios::package::cache::model_key("sel").unwrap();

        for tamper in ["embedded", "tensor", "envelope"] {
            let temp = tempfile::tempdir().unwrap();
            let model_root = temp.path().join("models");
            let options = HubOptions {
                revision: Some("tp1".into()),
                tensor_parallel: vec![1],
                model_cache: Some(model_root.clone()),
                ..Default::default()
            };
            let routes = package_routes(source_root, "sel/model", "tp1");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (first, cold) =
                counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
            worker.join().unwrap();
            let first_id = first.id().to_owned();
            let blobs = blob_count(&model_root);
            let entry_directory = first.directory().to_owned();

            match tamper {
                "embedded" => {
                    let locations = flare_helios::package::reader::Metadata::open(&entry_directory)
                        .unwrap()
                        .locations
                        .locations;
                    rewrite_payload_primary(
                        &entry_directory,
                        locations,
                        None,
                        &[("frontend.tokenizer", b"tampered")],
                    );
                }
                "tensor" => {
                    let source = flare_helios::package::cache::PackageSource::open(
                        &primary,
                        std::slice::from_ref(&publishers[0]),
                        flare_helios::package::protection::Selection::Deployment {
                            deployment: 0,
                            rank: None,
                        },
                    )
                    .unwrap();
                    let required: std::collections::BTreeSet<Vec<u8>> = source
                        .verified()
                        .iter()
                        .flat_map(|verified| {
                            flare_helios::package::reader::required_payloads(verified).unwrap()
                        })
                        .collect();
                    let mut locations =
                        flare_helios::package::reader::Metadata::open(&entry_directory)
                            .unwrap()
                            .locations
                            .locations;
                    let location = locations
                        .iter_mut()
                        .find(|location| {
                            !location.file.is_empty() && required.contains(&location.payload_id)
                        })
                        .expect("external required location");
                    location.tensor = "other".into();
                    rewrite_payload_primary(&entry_directory, locations, None, &[]);
                }
                "envelope" => {
                    std::fs::remove_file(entry_directory.join("model.safetensors")).unwrap();
                    std::fs::copy(&other_primary, entry_directory.join("model.safetensors"))
                        .unwrap();
                }
                _ => unreachable!(),
            }

            let routes = package_routes(source_root, "sel/model", "tp1");
            let (endpoint, worker, log) = mock_hub_counted(routes);
            let (repaired, warm) =
                counted_download_entry(&endpoint, "sel/model", &options, &publishers[0], &log);
            assert_ne!(
                repaired.id(),
                first_id,
                "{tamper}: corrupt entry was returned"
            );
            assert_eq!(
                flare_helios::package::reader::Metadata::open(repaired.directory())
                    .unwrap()
                    .envelope,
                source_envelope,
                "{tamper}: repaired entry changed signed semantics"
            );
            assert_eq!(
                flare_helios::package::cache::Root::open(&model_root)
                    .unwrap()
                    .ref_entry(&key)
                    .unwrap()
                    .as_deref(),
                Some(repaired.id()),
                "{tamper}: repaired entry was not selected"
            );
            let entry_metadata =
                flare_helios::package::reader::Metadata::open(repaired.directory()).unwrap();
            for location in &entry_metadata.locations.locations {
                let Some(source_location) = source_metadata
                    .locations
                    .locations
                    .iter()
                    .find(|source| source.payload_id == location.payload_id)
                else {
                    continue;
                };
                assert_eq!(
                    location_bytes(repaired.directory(), location),
                    location_bytes(source_root, source_location),
                    "{tamper}: payload {} changed",
                    flare_helios::package::PayloadId::from_bytes(
                        location.payload_id.as_slice().try_into().unwrap(),
                    )
                );
            }
            assert!(
                warm.files
                    .keys()
                    .all(|file| file.ends_with("/model.safetensors")),
                "{tamper}: warm repair downloaded dependency payloads: {warm:?}"
            );
            assert!(
                !warm.files.is_empty() || cold.files.is_empty(),
                "{tamper}: unexpected warm transfer accounting"
            );
            assert_eq!(
                blob_count(&model_root),
                blobs,
                "{tamper}: warm repair created new payload objects"
            );
            drop(first);
            drop(repaired);
            worker.join().unwrap();
        }
        assert_eq!(std::fs::read(&primary).unwrap(), source_primary);
    }

    #[test]
    fn network_progress_counts_a_partial_body_once_across_resume_retry() {
        use flare_helios::package::progress::AcquisitionProgress;
        use std::io::{BufRead, BufReader, Write};

        let body: Vec<u8> = (0..64u8).collect();
        let expected = body.clone();
        let half = body.len() / 2;
        let etag = "a".repeat(64);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let expected_path = format!("/owner/retry/resolve/{COMMIT}/model.safetensors");
        let worker = thread::spawn(move || {
            let mut body_attempt = 0;
            for _ in 0..4 {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with(&format!("GET {expected_path} ")), "{line}");
                let mut range = String::new();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.strip_prefix("range:") {
                        range = value.trim().to_owned();
                    }
                }
                if range == "bytes=0-0" {
                    socket
                        .write_all(&file_metadata(COMMIT, &etag, body.len()))
                        .unwrap();
                } else if range == "bytes=0-" {
                    body_attempt += 1;
                    let mut response = format!(
                        "HTTP/1.1 206 Partial Content\r\netag: {etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Range: bytes 0-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len() - 1,
                        body.len(),
                        body.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(&body[..half]);
                    socket.write_all(&response).unwrap();
                } else if range == format!("bytes={half}-") {
                    body_attempt += 1;
                    let mut response = format!(
                        "HTTP/1.1 206 Partial Content\r\netag: {etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Range: bytes {half}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len() - 1,
                        body.len(),
                        body.len() - half
                    )
                    .into_bytes();
                    response.extend_from_slice(&body[half..]);
                    socket.write_all(&response).unwrap();
                } else {
                    panic!("unexpected Range header {range:?}");
                }
            }
            assert_eq!(body_attempt, 2);
        });

        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let api = ApiBuilder::from_cache(cache.clone())
            .with_token(None)
            .with_endpoint(endpoint)
            .with_progress(false)
            .with_retries(1)
            .build()
            .unwrap();
        let repo = Repo::with_revision("owner/retry".into(), RepoType::Model, COMMIT.into());
        let progress = AcquisitionProgress::new();
        let path =
            fetch_file_with_progress(&api, &cache, &repo, "model.safetensors", Some(&progress))
                .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), expected);
        assert_eq!(progress.snapshot().network_bytes, expected.len() as u64);
    }

    #[test]
    fn network_progress_skips_the_resume_prefix() {
        use flare_helios::package::progress::AcquisitionProgress;
        use hf_hub::api::Progress as _;
        let progress = AcquisitionProgress::new();
        let mut counter = NetworkProgress {
            progress: progress.clone(),
            skip_next: true,
        };
        counter.init(10, "file");
        counter.update(3); // resume prefix, not this attempt's network bytes
        counter.update(4);
        counter.update(2);
        counter.finish();
        assert_eq!(progress.snapshot().network_bytes, 6);
    }

    #[test]
    fn multi_tp_entry_names_are_readable() {
        let name = acquisition_candidate(Some("gb202"), &[1, 2]).unwrap();
        assert!(name.starts_with("gb202-tp1+tp2-"), "{name}");
        assert!(!name.chars().all(|c| c.is_ascii_hexdigit()), "{name}");
        let single = acquisition_candidate(Some("gb10"), &[1]).unwrap();
        assert!(single.starts_with("gb10-tp1-"), "{single}");
        let deduplicated = acquisition_candidate(Some("gb202"), &[2, 1, 2]).unwrap();
        assert!(
            deduplicated.starts_with("gb202-tp1+tp2-"),
            "shared naming must deduplicate and sort union labels: {deduplicated}"
        );
        let unknown = acquisition_candidate(None, &[2, 1]).unwrap();
        assert!(
            unknown.starts_with("unknown-device-tp1+tp2-"),
            "signed-device fallback must use the shared unknown-device union: {unknown}"
        );
        assert!(acquisition_candidate(Some("gb202"), &[]).is_err());
        assert!(acquisition_candidate(Some("gb202"), &[0]).is_err());
    }

    fn inverse_tp_package(
        directory: &Path,
        canonical: &[u8],
        derived: &[u8],
    ) -> flare_helios::runtime_package::Publisher {
        use flare_helios::package::{PayloadId, conversion, derivation, export, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        std::fs::create_dir_all(directory).unwrap();
        let graph = directory.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let canonical_path = directory.join("payload-canonical");
        std::fs::write(&canonical_path, canonical).unwrap();
        let derived_path = directory.join("payload-derived");
        std::fs::write(&derived_path, derived).unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Inverse".into(),
            namespace: "inverse".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let payload = |relative: &str, path: &Path, bytes: &[u8]| export::PublishPayload {
            role: wire::PayloadRole::Weights,
            relative: relative.into(),
            source: export::pin_file(path, bytes.len() as u64).unwrap(),
            id: PayloadId::of(bytes),
            tensor: None,
        };
        let rank = |rank: u32,
                    deployment: &str,
                    target: PayloadId,
                    nbytes: u64,
                    payloads: Vec<export::PublishPayload>| export::PublishRank {
            body: wire::RankBody {
                rank,
                deployment_id: deployment.into(),
                context: Some(Default::default()),
                allocations: vec![wire::Allocation {
                    name: "weights".into(),
                    nbytes: 32,
                    alignment: 4096,
                    address: Some(0x1000_0000),
                    memory: flare_helios::runtime_package::wire::Memory::Device as i32,
                    ..Default::default()
                }],
                initializers: vec![wire::Initializer {
                    allocation: "weights".into(),
                    destination_offset: 0,
                    source: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: target.as_bytes().to_vec(),
                            offset: 0,
                            nbytes,
                        }),
                        role: wire::PayloadRole::Weights as i32,
                    }),
                }],
                graphs: vec![wire::Graph {
                    name: "decode".into(),
                    resource: 0,
                    node_count: 1,
                    ..Default::default()
                }],
                ..Default::default()
            },
            payloads,
            resources: vec![export::PublishResource {
                kind: wire::ExecutableKind::Graph,
                path: graph.clone(),
                nbytes: 11,
            }],
        };
        let canonical_id = PayloadId::of(canonical);
        let derived_id = PayloadId::of(derived);
        let request = export::PublishRequest {
            package: wire::PackageBody {
                package_id: vec![19; 32],
                model_identity: "inverse/tp".into(),
                hardware_identity: "sm120".into(),
                publisher: Some(wire::PublisherMetadata {
                    namespace: publisher.namespace.clone(),
                    official_source: "https://huggingface.co/inverse/tp".into(),
                    ..Default::default()
                }),
                frontend: Some(Default::default()),
                sampling_defaults: Some(Default::default()),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![
                export::PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "gb10-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![rank(
                        0,
                        "gb10-tp1",
                        canonical_id,
                        canonical.len() as u64,
                        vec![payload(
                            "weights/canonical.safetensors",
                            &canonical_path,
                            canonical,
                        )],
                    )],
                    payloads: vec![],
                },
                export::PublishDeployment {
                    body: wire::DeploymentBody {
                        deployment_id: "gb10-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(Default::default()),
                        ..Default::default()
                    },
                    ranks: vec![
                        rank(
                            0,
                            "gb10-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                        rank(
                            1,
                            "gb10-tp2",
                            derived_id,
                            derived.len() as u64,
                            vec![payload(
                                "weights/derived-rank1.safetensors",
                                &derived_path,
                                derived,
                            )],
                        ),
                    ],
                    payloads: vec![],
                },
            ],
            package_id: [19; 32],
            public_key: publisher.public_key,
            accepted: publisher.namespace.clone(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(directory)).unwrap();
        conversion::emit_derivations(
            directory,
            &[derivation::Derivation {
                target: derived_id,
                length: derived.len() as u64,
                operations: vec![
                    derivation::Operation::Copy(derivation::Copy {
                        source: canonical_id,
                        source_offset: 0,
                        destination_offset: 8,
                        contiguous: 8,
                        outer: Vec::new(),
                    }),
                    derivation::Operation::Copy(derivation::Copy {
                        source: canonical_id,
                        source_offset: 8,
                        destination_offset: 0,
                        contiguous: 8,
                        outer: Vec::new(),
                    }),
                ],
            }],
            &derivation::Limits::default(),
        )
        .unwrap();
        publisher
    }

    #[test]
    fn tp2_target_cache_recovers_tp1_over_http_without_canonical_transfer() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("package");
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived = canonical[8..16].to_vec();
        derived.extend_from_slice(&canonical[..8]);
        let publisher = inverse_tp_package(&package, &canonical, &derived);
        let model_root = temp.path().join("models");
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let derived_id = flare_helios::package::PayloadId::of(&derived);
        store
            .insert_payload(derived_id, derived.len() as u64, derived.as_slice())
            .unwrap();
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (log, entry_id) =
            counted_download(&package, "inverse/tp", "gb10", &options, &publisher);
        assert!(entry_id.starts_with("gb10-tp1-"), "{entry_id}");
        assert!(
            !log.files
                .keys()
                .any(|path| path.contains("weights/canonical")),
            "canonical payload was transferred: {log:?}"
        );
        assert!(
            !log.files
                .keys()
                .any(|path| path.contains("weights/derived")),
            "cached TP2 target was transferred again: {log:?}"
        );
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_fixture_cold_and_warm_use_friendly_names_and_default() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let profile = package.join("publisher.json");
        assert!(
            package.join("model.safetensors").is_file(),
            "legacy fixture lacks model.safetensors: {}",
            package.display()
        );
        assert!(
            profile.is_file(),
            "legacy fixture lacks publisher.json: {}",
            package.display()
        );
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (cold, cold_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        assert!(
            cold_id.starts_with("unknown-device-tp2-"),
            "legacy missing-device name is not explicit: {cold_id}"
        );
        assert!(!cold_id.contains("sm120"), "{cold_id}");
        assert!(
            cold.files.keys().any(|path| path.contains("model-data-")),
            "cold legacy download did not transfer companions: {cold:?}"
        );

        let (warm, warm_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        assert_eq!(
            warm_id, cold_id,
            "warm legacy download did not reuse the named entry"
        );
        assert!(
            !warm.files.keys().any(|path| path.contains("model-data-")),
            "warm legacy download transferred companions: {warm:?}"
        );
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(warm_id.as_str()),
            "warm legacy download did not update the default"
        );
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_historical_hash_warm_validates_signed_tp() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let primary = package.join("model.safetensors");
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(package.join("publisher.json")).unwrap())
                .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let primary_bytes = std::fs::read(&primary).unwrap();
        let old_id = format!(
            "{}-legacy",
            flare_helios::package::PayloadId::of(&primary_bytes)
        );
        store.import_legacy_staged(&primary, &key, &old_id).unwrap();

        let exact = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        let warm = legacy_warm_entry(
            &primary,
            &exact,
            "legacy/cluster",
            "custom-branch",
            COMMIT,
            "model.safetensors",
            1,
            &publishers,
            &model_root,
        )
        .unwrap();
        assert_eq!(warm.unwrap().id(), old_id);

        for requested in [vec![4], vec![1, 2]] {
            let wrong = HubOptions {
                tensor_parallel: requested,
                ..Default::default()
            };
            let warm = legacy_warm_entry(
                &primary,
                &wrong,
                "legacy/cluster",
                "custom-branch",
                COMMIT,
                "model.safetensors",
                1,
                &publishers,
                &model_root,
            )
            .unwrap();
            assert!(
                warm.is_none(),
                "historical hash warm accepted unsupported TP {:?}",
                wrong.tensor_parallel
            );
        }
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with model.safetensors and publisher.json"]
    fn legacy_warm_rejects_mismatched_explicit_tp_without_default_change() {
        let package = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name a signed legacy package directory"),
        );
        let profile = package.join("publisher.json");
        let publishers: Vec<flare_helios::runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![2],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "legacy/cluster",
            "custom-branch",
            &options,
            &publishers[0],
        );
        let key = flare_helios::package::cache::model_key("legacy/cluster").unwrap();
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(entry_id.as_str())
        );

        let mut wrong = options.clone();
        wrong.tensor_parallel = vec![1, 2];
        let warm = legacy_warm_entry(
            &package.join("model.safetensors"),
            &wrong,
            "legacy/cluster",
            "custom-branch",
            COMMIT,
            "model.safetensors",
            1,
            &publishers,
            &model_root,
        )
        .unwrap();
        assert!(
            warm.is_none(),
            "legacy warm path accepted a mismatched explicit TP"
        );
        assert!(
            install_ready(
                "legacy/cluster",
                &package.join("model.safetensors"),
                &wrong,
                &model_root,
                1,
                &publishers,
            )
            .is_err(),
            "legacy cold install accepted a mismatched explicit TP"
        );
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(entry_id.as_str()),
            "failed TP mismatch changed the default"
        );
    }

    #[test]
    fn custom_revision_uses_signed_device_identity() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "owner/offline",
            "custom-branch",
            &options,
            &publisher,
        );
        assert!(
            entry_id.starts_with("gb10-tp1-"),
            "custom revision must use the signed gb10 identity, got {entry_id}"
        );
    }

    #[test]
    fn signed_custom_cluster_id_uses_package_hardware_identity() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) =
            signed_payload_fixture_with(temp.path(), "cluster-west", "gb10");
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("custom-branch".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) = counted_download(
            &package,
            "owner/offline",
            "custom-branch",
            &options,
            &publisher,
        );
        assert!(
            entry_id.starts_with("gb10-tp1-"),
            "custom cluster ID must use signed gb10 hardware, got {entry_id}"
        );
    }

    #[test]
    fn old_sm_only_package_gets_unknown_device_label() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) =
            signed_payload_fixture_with(temp.path(), "sm120-tp1", "sm120");
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb202".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (_log, entry_id) =
            counted_download(&package, "owner/offline", "gb202", &options, &publisher);
        assert!(
            entry_id.starts_with("unknown-device-tp1-"),
            "old SM-only package must get an explicit unknown-device label, got {entry_id}"
        );
        assert!(!entry_id.contains("sm120"), "{entry_id}");
        assert!(!entry_id.starts_with("gb202"), "{entry_id}");
    }

    #[test]
    fn cold_and_warm_downloads_report_actual_transfers() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            ..Default::default()
        };
        let (cold, cold_id) =
            counted_download(&package, "owner/offline", "gb10", &options, &publisher);
        assert!(
            cold.files
                .keys()
                .any(|name| name.contains("weights/layer-000.safetensors")),
            "{cold:?}"
        );
        assert!(cold.bytes > 0, "{cold:?}");
        assert!(cold_id.starts_with("gb10-tp1-"), "{cold_id}");
        assert!(
            !cold_id.chars().all(|c| c.is_ascii_hexdigit()),
            "entry name must not be a package hash: {cold_id}"
        );
        let selector = source_selector("owner/offline", "gb10", Some("model.safetensors"), &[1]);
        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert!(
            store.read_source_ref(&selector).unwrap().is_some(),
            "cold download did not record its source selector"
        );

        let (warm, warm_id) =
            counted_download(&package, "owner/offline", "gb10", &options, &publisher);
        assert_eq!(
            warm_id, cold_id,
            "warm download did not reuse the ready entry"
        );
        assert!(
            !warm.files.keys().any(|name| name.contains("/weights/")),
            "warm download transferred companions {warm:?}"
        );
        assert!(
            !warm.files.keys().any(|name| name.contains("deployments/")),
            "warm download transferred executables {warm:?}"
        );
        let key = flare_helios::package::cache::model_key("owner/offline").unwrap();
        assert_eq!(
            store.ref_entry(&key).unwrap().as_deref(),
            Some(warm_id.as_str()),
            "warm download did not update the default after success"
        );
    }

    /// Serve a scripted sequence that tolerates repeated metadata calls: after
    /// the first two requests for a path, the body response is used.
    fn mock_hub_flexible(
        routes: std::collections::HashMap<String, (Vec<u8>, Vec<u8>)>,
        _expected: usize,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            let mut seen: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            let mut idle = 0;
            let mut served = 0;
            loop {
                let (mut socket, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if served > 0 {
                            idle += 1;
                            // Stop after a quiet period once the client is done;
                            // unexpected paths still panic through the route table.
                            if idle > 200 {
                                break;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("accept failed: {error}"),
                };
                idle = 0;
                served += 1;
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .expect("request path")
                    .to_owned();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                let (metadata, body) = routes.get(&path).unwrap_or_else(|| {
                    panic!("unexpected request {path}");
                });
                let count = seen.entry(path).or_insert(0);
                *count += 1;
                let response = if *count <= 2 { metadata } else { body };
                socket.write_all(response).unwrap();
            }
        });
        (endpoint, worker)
    }

    fn metadata_response_with(files: &[&str]) -> Vec<u8> {
        let siblings: Vec<String> = files
            .iter()
            .map(|name| format!("{{\"rfilename\":\"{name}\"}}"))
            .collect();
        let body = format!(
            "{{\"sha\":\"{COMMIT}\",\"siblings\":[{}]}}",
            siblings.join(",")
        );
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\netag: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            "0".repeat(64),
            body.len()
        )
        .into_bytes()
    }

    fn metadata_response() -> Vec<u8> {
        revision_response(COMMIT)
    }

    fn revision_response(commit: &str) -> Vec<u8> {
        let body =
            format!(r#"{{"sha":"{commit}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#);
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\netag: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            "0".repeat(64),
            body.len()
        ).into_bytes()
    }

    fn file_metadata(commit: &str, etag: &str, size: usize) -> Vec<u8> {
        format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: {commit}\r\netag: {etag}\r\nContent-Range: bytes 0-0/{size}\r\nContent-Length: 1\r\nConnection: close\r\n\r\np").into_bytes()
    }

    fn file_body(etag: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 206 Partial Content\r\netag: {etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Range: bytes 0-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len().saturating_sub(1),
            body.len(),
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn file_response(body: &[u8]) -> Vec<u8> {
        [
            format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes(),
            body.to_vec(),
        ]
        .concat()
    }

    #[cfg(unix)]
    fn assert_same_file(left: &Path, right: &Path) {
        use std::os::unix::fs::MetadataExt;
        let left = std::fs::metadata(left).unwrap();
        let right = std::fs::metadata(right).unwrap();
        assert_eq!((left.dev(), left.ino()), (right.dev(), right.ino()));
    }

    fn test_api(cache: &Cache, endpoint: String) -> Api {
        ApiBuilder::from_cache(cache.clone())
            .with_token(None)
            .with_endpoint(endpoint)
            .with_progress(false)
            .build()
            .unwrap()
    }

    /// Two deployments with derivations; a selecting TP1 must fetch only TP1's
    /// dependency source, never the unselected TP2 source or embedded frontend.
    fn selected_tp_package() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        Vec<flare_helios::runtime_package::Publisher>,
    ) {
        use flare_helios::package::{
            PayloadId, container, conversion, derivation, export, schema::wire,
        };
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let scratch = tempfile::tempdir().unwrap();
        let directory = scratch.path().join("package");
        std::fs::create_dir_all(&directory).unwrap();
        let s1 = b"tp1 canonical source".to_vec();
        let t1 = s1[..s1.len() - 3].to_vec();
        let s2 = b"tp2 canonical source".to_vec();
        let t2 = s2[..s2.len() - 3].to_vec();
        let tokenizer = b"tokenizer bytes".to_vec();
        let (s1_id, t1_id) = (PayloadId::of(&s1), PayloadId::of(&t1));
        let (s2_id, t2_id) = (PayloadId::of(&s2), PayloadId::of(&t2));
        let tokenizer_id = PayloadId::of(&tokenizer);
        let write = |name: &str, bytes: &[u8]| {
            let path = scratch.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let s1_path = write("s1.bin", &s1);
        let t1_path = write("t1.bin", &t1);
        let s2_path = write("s2.bin", &s2);
        let t2_path = write("t2.bin", &t2);
        let tokenizer_path = write("tokenizer.bin", &tokenizer);
        let graph = write("graph.bin", b"graph-bytes");
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Sel".into(),
            namespace: "sel".into(),
            public_key,
            development: false,
        };
        let payload = |id: PayloadId, path: &std::path::Path, nbytes: usize, relative: &str| {
            export::PublishPayload {
                role: wire::PayloadRole::Weights,
                relative: relative.into(),
                tensor: Some(container::COMPANION_TENSOR.into()),
                source: container::Slice::new(
                    std::sync::Arc::new(std::fs::File::open(path).unwrap()),
                    0,
                    nbytes as u64,
                )
                .unwrap(),
                id,
            }
        };
        let contract = |id: &str| flare_helios::runtime_package::wire::ServingContract {
            model_id: "sel".into(),
            identity: format!("sel-{id}"),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = |deployment: &str, rank: u32, target: PayloadId, nbytes: usize| wire::RankBody {
            rank,
            deployment_id: deployment.into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: nbytes as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: target.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: nbytes as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "sel".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "sel".into(),
                    official_source: "https://huggingface.co/sel/model".into(),
                    ..Default::default()
                }),
                assets: vec![wire::FrontendAsset {
                    name: "tokenizer".into(),
                    payload: Some(wire::PayloadUse {
                        slice: Some(wire::PayloadSlice {
                            payload_id: tokenizer_id.as_bytes().to_vec(),
                            offset: 0,
                            nbytes: tokenizer.len() as u64,
                        }),
                        role: wire::PayloadRole::Frontend as i32,
                    }),
                    sha256: flare_helios::package::protection::sha256(&tokenizer).to_vec(),
                }],
                ..Default::default()
            },
            payloads: vec![export::PublishPayload {
                role: wire::PayloadRole::Frontend,
                relative: String::new(),
                tensor: Some("frontend.tokenizer".into()),
                source: container::Slice::new(
                    std::sync::Arc::new(std::fs::File::open(&tokenizer_path).unwrap()),
                    0,
                    tokenizer.len() as u64,
                )
                .unwrap(),
                id: tokenizer_id,
            }],
            deployments: vec![
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp1".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 1,
                        contract: Some(contract("tp1")),
                        ..Default::default()
                    },
                    ranks: vec![export::PublishRank {
                        body: rank("sm120-tp1", 0, t1_id, t1.len()),
                        payloads: vec![
                            payload(s1_id, &s1_path, s1.len(), "weights/tp1-source.safetensors"),
                            payload(t1_id, &t1_path, t1.len(), "weights/tp1-target.safetensors"),
                        ],
                        resources: vec![export::PublishResource {
                            kind: wire::ExecutableKind::Graph,
                            path: graph.clone(),
                            nbytes: 11,
                        }],
                    }],
                },
                export::PublishDeployment {
                    payloads: vec![],
                    body: wire::DeploymentBody {
                        deployment_id: "sm120-tp2".into(),
                        architecture: vec![12, 0],
                        tensor_parallel: 2,
                        contract: Some(contract("tp2")),
                        ..Default::default()
                    },
                    ranks: vec![
                        export::PublishRank {
                            body: rank("sm120-tp2", 0, t2_id, t2.len()),
                            payloads: vec![
                                payload(
                                    s2_id,
                                    &s2_path,
                                    s2.len(),
                                    "weights/tp2-source.safetensors",
                                ),
                                payload(
                                    t2_id,
                                    &t2_path,
                                    t2.len(),
                                    "weights/tp2-target.safetensors",
                                ),
                            ],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph.clone(),
                                nbytes: 11,
                            }],
                        },
                        export::PublishRank {
                            body: wire::RankBody {
                                rank: 1,
                                deployment_id: "sm120-tp2".into(),
                                context: Some(Default::default()),
                                allocations: vec![wire::Allocation {
                                    name: "device-1".into(),
                                    nbytes: t2.len() as u64,
                                    alignment: 4096,
                                    ..Default::default()
                                }],
                                graphs: vec![wire::Graph {
                                    name: "decode".into(),
                                    resource: 0,
                                    node_count: 1,
                                    ..Default::default()
                                }],
                                ..Default::default()
                            },
                            payloads: vec![],
                            resources: vec![export::PublishResource {
                                kind: wire::ExecutableKind::Graph,
                                path: graph,
                                nbytes: 11,
                            }],
                        },
                    ],
                },
            ],
            package_id: [43u8; 32],
            public_key,
            accepted: "sel".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&directory)).unwrap();
        let recipe = |target: PayloadId, source: PayloadId, length: usize| derivation::Derivation {
            target,
            length: length as u64,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: 0,
                destination_offset: 0,
                contiguous: length as u64,
                outer: Vec::new(),
            })],
        };
        conversion::emit_derivations(
            &directory,
            &[
                recipe(t1_id, s1_id, t1.len()),
                recipe(t2_id, s2_id, t2.len()),
            ],
            &derivation::Limits::default(),
        )
        .unwrap();
        (
            scratch,
            directory.join("model.safetensors"),
            vec![publisher],
        )
    }

    #[test]
    fn selected_tp_fetches_only_its_own_dependency_companions() {
        let (_scratch, primary, publishers) = selected_tp_package();
        let options = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let files = dependency_files(&primary, &options, 1, &publishers).unwrap();
        assert_eq!(
            files,
            vec![
                "deployments/sm120-tp1/rank-0/graphs.safetensors".to_string(),
                "weights/tp1-source.safetensors".to_string(),
            ],
            "selected TP fetches its own executables and dependency source only"
        );
        assert!(!files.iter().any(|file| file.contains("tp2")), "{files:?}");
        assert!(
            !files.iter().any(|file| file.contains("tokenizer")),
            "{files:?}"
        );
    }

    #[test]
    fn payload_download_fetches_only_the_selected_closure_over_http() {
        use flare_helios::package::{PayloadId, container, export, reader, schema::wire};
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().join("hub"));
        let source = temp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        let weights = b"selected closure weights".repeat(4);
        let id = PayloadId::of(&weights);
        let raw = temp.path().join("weights.bin");
        std::fs::write(&raw, &weights).unwrap();
        let graph = temp.path().join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key,
            development: false,
        };
        let runtime = flare_helios::runtime_package::wire::ServingContract {
            model_id: "closure".into(),
            identity: "closure-sm120".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(flare_helios::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![flare_helios::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "gb10-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weights.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: weights.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "owner/closure".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(flare_helios::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    flare_helios::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "test".into(),
                    official_source: "https://huggingface.co/test/closure".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: "gb10-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(runtime),
                    ..Default::default()
                },
                ranks: vec![export::PublishRank {
                    body: rank,
                    payloads: vec![export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: flare_helios::package::container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                            0,
                            (weights.len() as u64) as u64,
                        )
                        .unwrap(),
                        id,
                    }],
                    resources: vec![export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: (11) as u64,
                    }],
                }],
            }],
            package_id: [17u8; 32],
            public_key,
            accepted: "test".into(),
            no_weights: false,
        };
        export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();
        let metadata = reader::Metadata::open(&source).unwrap();
        let files: Vec<String> = std::iter::once("model.safetensors".to_string())
            .chain(
                metadata
                    .locations
                    .locations
                    .iter()
                    .filter(|location| !location.file.is_empty())
                    .map(|location| location.file.clone()),
            )
            .collect();

        // The snapshot and blob cache are regular files on the same volume. Hard
        // links exercise the same content/closure contract without requiring
        // Windows symlink privilege.
        let snapshot = temp.path().join("snapshot");
        std::fs::create_dir_all(snapshot.join("blobs")).unwrap();
        for name in &files {
            let blob = snapshot.join("blobs").join(name.replace('/', "_"));
            std::fs::copy(source.join(name), &blob).unwrap();
            let link = snapshot.join(name);
            std::fs::create_dir_all(link.parent().unwrap()).unwrap();
            std::fs::hard_link(&blob, &link).unwrap();
        }
        let decoy = snapshot.join("weights/unselected.safetensors");
        std::fs::write(&decoy, b"decoy").unwrap();

        let repo = Repo::with_revision("owner/closure".into(), RepoType::Model, COMMIT.into());
        let resolve = |name: &str| format!("/owner/closure/resolve/{COMMIT}/{name}");
        let mut responses: Vec<(String, Vec<u8>)> = vec![(
            format!("/api/models/owner/closure/revision/{COMMIT}"),
            metadata_response_with(&[
                "model.safetensors",
                files[1].as_str(),
                "weights/unselected.safetensors",
            ]),
        )];
        for name in &files {
            use sha2::{Digest, Sha256};
            let body = std::fs::read(source.join(name)).unwrap();
            let etag = format!("{:x}", Sha256::digest(&body));
            responses.push((resolve(name), file_metadata(COMMIT, &etag, body.len())));
            responses.push((
                resolve(name),
                {
                    let mut response = format!(
                        "HTTP/1.1 206 Partial Content\r\netag: {etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Range: bytes 0-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len().saturating_sub(1),
                        body.len(),
                        body.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(&body);
                    response
                },
            ));
        }
        let _ = responses;
        let mut routes = std::collections::HashMap::new();
        routes.insert(
            format!("/api/models/owner/closure/revision/{COMMIT}"),
            (
                metadata_response_with(&[
                    "model.safetensors",
                    files[1].as_str(),
                    "weights/unselected.safetensors",
                ]),
                Vec::new(),
            ),
        );
        for name in &files {
            use sha2::{Digest, Sha256};
            let body = std::fs::read(source.join(name)).unwrap();
            let etag = format!("{:x}", Sha256::digest(&body));
            let metadata = file_metadata(COMMIT, &etag, body.len());
            let mut response = format!(
                "HTTP/1.1 206 Partial Content\r\netag: {etag}\r\nx-repo-commit: {COMMIT}\r\nContent-Range: bytes 0-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len().saturating_sub(1),
                body.len(),
                body.len()
            )
            .into_bytes();
            response.extend_from_slice(&body);
            routes.insert(resolve(name), (metadata, response));
        }
        // Two requests per file plus the revision metadata request.
        let (endpoint, worker) = mock_hub_snapshot(routes);
        let api = test_api(&cache, endpoint);
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb10".into()),
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let (primary, _pinned) = fetch_primary(&api, &cache, &repo, None).unwrap();
        for name in dependency_files(&primary, &options, 1, &[publisher.clone()]).unwrap() {
            fetch_dependency(
                &api,
                &cache,
                &Repo::with_revision(repo.url(), RepoType::Model, COMMIT.into()),
                &primary,
                &name,
            )
            .unwrap();
        }
        worker.join().unwrap();
        // Only the selected closure was requested: the decoy is not in the snapshot.
        assert!(
            !cache
                .path()
                .join(repo.folder_name())
                .join(COMMIT)
                .join("weights/unselected.safetensors")
                .exists()
        );
        assert!(
            !cache
                .path()
                .join(repo.folder_name())
                .join("flare-packages")
                .exists()
        );
        let entry = install_ready(
            "owner/closure",
            &primary,
            &options,
            &model_root,
            1,
            &[publisher.clone()],
        )
        .unwrap();
        assert!(
            entry
                .directory()
                .join("weights/layer-000.safetensors")
                .is_file()
        );
        // Repository names need not match the model identity in signed metadata.
        let expected_envelope = authenticated_source_envelope(
            &primary,
            &options,
            1,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        remember_download(
            &model_root,
            "mirror/closure",
            "gb202",
            COMMIT,
            "model.safetensors",
            &options,
            1,
            &entry,
            expected_envelope,
        )
        .unwrap();
        std::fs::remove_dir_all(cache.path()).unwrap();
        let offline = HubOptions {
            offline: true,
            revision: Some("gb202".into()),
            model_cache: Some(model_root.clone()),
            token_file: Some(temp.path().join("nonexistent-token")),
            ..options.clone()
        };
        for revision in ["gb202", COMMIT] {
            for filename in [None, Some("model.safetensors")] {
                let requested = HubOptions {
                    filename: filename.map(str::to_owned),
                    ..offline.clone()
                };
                let hit =
                    offline_entry(&model_root, "mirror/closure", revision, &requested, 1).unwrap();
                assert_eq!(hit.directory(), entry.directory());
                verify_offline_entry(&hit, &requested, 1, &[publisher.clone()]).unwrap();
                assert!(verify_offline_entry(&hit, &requested, 1, &[]).is_err());
            }
        }
        assert!(!cache.path().exists());
        let wrong_file = HubOptions {
            filename: Some("other.safetensors".into()),
            ..offline.clone()
        };
        assert!(offline_entry(&model_root, "mirror/closure", "gb202", &wrong_file, 1).is_err());
        assert!(offline_entry(&model_root, "mirror/closure", "gb10", &offline, 1).is_err());
        let wrong_tp = HubOptions {
            tensor_parallel: vec![2],
            ..offline.clone()
        };
        let hit = offline_entry(&model_root, "mirror/closure", "gb202", &wrong_tp, 1).unwrap();
        assert!(verify_offline_entry(&hit, &wrong_tp, 1, &[publisher.clone()]).is_err());
        drop(hit);

        let payload = entry.directory().join("weights/layer-000.safetensors");
        let mut permissions = std::fs::metadata(&payload).unwrap().permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(&payload, permissions).unwrap();
        let mut corrupted = std::fs::read(&payload).unwrap();
        *corrupted.last_mut().unwrap() ^= 1;
        std::fs::write(&payload, corrupted).unwrap();
        assert!(verify_offline_entry(&entry, &offline, 1, &[publisher]).is_err());

        let store = flare_helios::package::cache::Root::open(&model_root).unwrap();
        assert!(store.evict(entry.model(), entry.id()).is_err());
        let (model, id) = (entry.model().to_owned(), entry.id().to_owned());
        drop(entry);
        store.evict(&model, &id).unwrap();
        assert!(offline_entry(&model_root, "mirror/closure", "gb202", &offline, 1).is_err());
    }

    #[test]
    fn transport_staging_is_operation_owned_and_cleaned() {
        let temp = tempfile::tempdir().unwrap();
        let first = TransportStaging::new(temp.path(), None).unwrap();
        let second = TransportStaging::new(temp.path(), None).unwrap();
        assert_ne!(first.path(), second.path());
        let first_path = first.path().to_owned();
        let second_path = second.path().to_owned();
        assert!(first_path.is_dir() && second_path.is_dir());
        drop(first);
        assert!(!first_path.exists());
        assert!(second_path.is_dir());
        drop(second);
        assert!(!second_path.exists());
        assert!(temp.path().join("staging/trees").is_dir());
        let explicit = temp.path().join("explicit");
        let third = TransportStaging::new(temp.path(), Some(&explicit)).unwrap();
        let third_path = third.path().to_owned();
        assert!(third_path.starts_with(&explicit));
        drop(third);
        assert!(!third_path.exists());
    }

    #[test]
    fn requested_rank_count_mismatch_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, _commit) = signed_payload_fixture(temp.path());
        let verified = flare_helios::runtime_package::VerifiedPackage::open(
            &package,
            std::slice::from_ref(&publisher),
            None,
        )
        .unwrap();
        assert!(enforce_legacy_tensor_parallel(&verified, &[1]).is_ok());
        let error = enforce_legacy_tensor_parallel(&verified, &[2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("requested [2]"), "{error}");
        let error = enforce_legacy_tensor_parallel(&verified, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("requested [1, 2]"), "{error}");
    }

    #[test]
    fn offline_download_never_imports_hf_snapshots() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().join("hub"));
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[])).unwrap();
        cache.repo(repo).create_ref(COMMIT).unwrap();
        let options = HubOptions {
            offline: true,
            revision: Some("gb202".into()),
            model_cache: Some(temp.path().join("flare")),
            token_file: Some(temp.path().join("nonexistent-token")),
            ..Default::default()
        };
        let error = download("owner/model", &options, 1, &[], None, &[])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "{error}"
        );
        assert_eq!(
            std::fs::read_dir(temp.path().join("flare/models"))
                .unwrap()
                .count(),
            0
        );
        std::fs::remove_dir_all(cache.path()).unwrap();
        let error = download("owner/model", &options, 1, &[], None, &[])
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "{error}"
        );
        assert!(!cache.path().exists());
    }

    #[test]
    fn offline_branch_does_not_select_an_older_commit_for_its_tp() {
        use flare_helios::package::cache::Root;
        let temp = tempfile::tempdir().unwrap();
        let root = Root::open(temp.path()).unwrap();
        let first = root.stage("model", "first").unwrap().commit().unwrap();
        let second = root.stage("model", "second").unwrap().commit().unwrap();
        let tp1 = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let tp2 = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp1,
            1,
            &first,
            [0; 32],
        )
        .unwrap();
        let other_commit = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            other_commit,
            "model.safetensors",
            &tp2,
            1,
            &second,
            [0; 32],
        )
        .unwrap();
        let branch = offline_entry(temp.path(), "owner/model", "gb202", &tp1, 1).unwrap();
        assert_eq!(branch.id(), second.id());
        let pinned = offline_entry(temp.path(), "owner/model", COMMIT, &tp1, 1).unwrap();
        assert_eq!(pinned.id(), first.id());
        drop(pinned);
        drop(branch);
        let first_id = first.id().to_owned();
        drop(first);
        root.evict("model", &first_id).unwrap();
        assert_eq!(
            offline_entry(temp.path(), "owner/model", "gb202", &tp1, 1)
                .unwrap()
                .id(),
            second.id()
        );
        assert!(offline_entry(temp.path(), "owner/model", COMMIT, &tp1, 1).is_err());
    }

    #[test]
    fn offline_download_selects_an_already_installed_nondefault_entry() {
        use flare_helios::package::cache::{Root, model_key};
        let temp = tempfile::tempdir().unwrap();
        let (package, _profile, publisher, commit) = signed_payload_fixture(temp.path());
        let model_root = temp.path().join("models");
        let options = HubOptions {
            revision: Some("gb202".into()),
            tensor_parallel: vec![1],
            model_cache: Some(model_root.clone()),
            offline: true,
            ..Default::default()
        };
        let key = model_key("owner/offline").unwrap();
        let root = Root::open(&model_root).unwrap();
        let prior = root.stage_named(&key, "prior").unwrap().commit().unwrap();
        root.publish_entry(&prior).unwrap();

        // Install the signed entry but leave the prior default selected.
        let installed = install_ready(
            "owner/offline",
            &package,
            &options,
            &model_root,
            1,
            &[publisher.clone()],
        )
        .unwrap();
        let installed_id = installed.id().to_owned();
        assert_ne!(
            root.ref_entry(&key).unwrap().as_deref(),
            Some(installed_id.as_str())
        );
        let expected_envelope = authenticated_source_envelope(
            &package,
            &options,
            1,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        remember_download(
            &model_root,
            "owner/offline",
            "gb202",
            &commit,
            "model.safetensors",
            &options,
            1,
            &installed,
            expected_envelope,
        )
        .unwrap();
        drop(installed);

        // The public offline download path resolves the installed source reference
        // and selects that existing entry without installing or downloading again.
        let selected = download_with_token(
            "owner/offline@gb202",
            &options,
            1,
            None,
            &[],
            None,
            std::slice::from_ref(&publisher),
        )
        .unwrap();
        assert_eq!(selected.id(), installed_id);
        assert_eq!(
            root.ref_entry(&key).unwrap().as_deref(),
            Some(installed_id.as_str())
        );
        assert!(root.entry(&key, &installed_id).is_ok());
        drop(selected);
        drop(prior);
        let _ = std::fs::remove_dir_all(temp.path());
    }

    #[test]
    fn failed_source_ref_recording_preserves_the_prior_default() {
        use flare_helios::package::cache::{Root, model_key};
        let temp = tempfile::tempdir().unwrap();
        let model_root = temp.path().join("models");
        let root = Root::open(&model_root).unwrap();
        let key = model_key("owner/closure").unwrap();
        // A previously acquired and selected version.
        let prior = root.stage_named(&key, "prior").unwrap().commit().unwrap();
        root.publish_entry(&prior).unwrap();
        // The newly installed but not yet selected entry, as install_ready leaves it.
        let fresh = root.stage_named(&key, "fresh").unwrap().commit().unwrap();
        // Force the first source-reference publish to fail: its destination is a directory.
        let options = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let selector = source_selector("owner/closure", "gb202", Some("model.safetensors"), &[1]);
        std::fs::create_dir_all(
            model_root
                .join("sources")
                .join(flare_helios::package::PayloadId::of(selector.as_bytes()).to_string()),
        )
        .unwrap();
        assert!(
            remember_download(
                &model_root,
                "owner/closure",
                "gb202",
                COMMIT,
                "model.safetensors",
                &options,
                1,
                &fresh,
                [0; 32],
            )
            .is_err()
        );
        // The prior default is untouched and the failed acquisition is not selected.
        assert_eq!(root.ref_entry(&key).unwrap().as_deref(), Some(prior.id()));
        assert_ne!(prior.id(), fresh.id());
        let fresh_id = fresh.id().to_owned();
        drop(fresh);
        root.remove(&key, &fresh_id, false).unwrap();
        let _ = std::fs::remove_dir_all(temp.path());
    }

    #[test]
    fn offline_exact_tp_entry_survives_eviction_of_the_latest_union() {
        use flare_helios::package::cache::Root;
        let temp = tempfile::tempdir().unwrap();
        let root = Root::open(temp.path()).unwrap();
        let tp1_entry = root.stage("model", "tp1").unwrap().commit().unwrap();
        let tp2_entry = root.stage("model", "tp2").unwrap().commit().unwrap();
        let tp1 = HubOptions {
            tensor_parallel: vec![1],
            ..Default::default()
        };
        let tp2 = HubOptions {
            tensor_parallel: vec![2],
            ..Default::default()
        };
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp1,
            1,
            &tp1_entry,
            [1; 32],
        )
        .unwrap();
        remember_download(
            temp.path(),
            "owner/model",
            "gb202",
            COMMIT,
            "model.safetensors",
            &tp2,
            1,
            &tp2_entry,
            [2; 32],
        )
        .unwrap();
        let tp2_id = tp2_entry.id().to_owned();
        drop(tp2_entry);
        root.evict("model", &tp2_id).unwrap();
        // The unqualified target is gone, but the last resolved commit is known
        // and the same-commit exact TP1 entry is still ready.
        let (hit, expected) =
            offline_entry_with_expected(temp.path(), "owner/model", "gb202", &tp1, 1).unwrap();
        assert_eq!(hit.id(), tp1_entry.id());
        assert_eq!(expected, [1; 32]);
    }

    #[test]
    fn download_resolves_hardware_branch_then_fetches_the_exact_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let resolve = format!("/owner/model/resolve/{COMMIT}/model.safetensors");
        let package = fixture(&[]);
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb202".into(),
                metadata_response(),
            ),
            (
                resolve.clone(),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
            (
                resolve.clone(),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
            (resolve, file_response(&package)),
        ]);
        let api = test_api(&cache, endpoint);
        let path = fetch_primary(&api, &cache, &repo, None).unwrap().0;
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), package);
    }

    #[test]
    fn online_cache_hit_resolves_revision_without_redownloading_weights() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), &fixture(&[])).unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/owner/model/revision/gb10".into(),
            metadata_response(),
        )]);
        let path = fetch_primary(&test_api(&cache, endpoint), &cache, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture(&[]));
    }
    #[test]
    fn companions_download_from_the_primary_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = "model-data-01-of-01.safetensors";
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        let etag = "b".repeat(64);
        let mut routes = std::collections::HashMap::new();
        routes.insert(
            "/api/models/owner/model/revision/gb202".into(),
            (metadata_response(), Vec::new()),
        );
        routes.insert(
            format!("/owner/model/resolve/{COMMIT}/{name}"),
            (file_metadata(COMMIT, &etag, 4), file_body(&etag, b"data")),
        );
        let (endpoint, worker) = mock_hub_flexible(routes, 4);
        let api = test_api(&cache, endpoint);
        let (path, pinned) = fetch_primary(&api, &cache, &repo, None).unwrap();
        fetch_dependency(&api, &cache, &pinned, &path, name).unwrap();
        worker.join().unwrap();
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            b"data"
        );
    }

    #[test]
    fn cached_primary_blob_is_linked_without_downloading_its_body() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let package = fixture(&[]);
        std::fs::write(&blob, &package).unwrap();
        let (endpoint, worker) = mock_hub(vec![
            (
                "/api/models/owner/model/revision/gb10".into(),
                metadata_response(),
            ),
            (
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                file_metadata(COMMIT, "fixture", package.len()),
            ),
        ]);
        let path = fetch_primary(&test_api(&cache, endpoint), &cache, &repo, None)
            .unwrap()
            .0;
        worker.join().unwrap();
        assert_eq!(
            path,
            snapshot(&cache, &repo, COMMIT).join("model.safetensors")
        );
        assert_eq!(std::fs::read(&path).unwrap(), package);
        #[cfg(unix)]
        assert_same_file(&blob, &path);
    }

    #[test]
    fn revisions_share_downloaded_payload_without_replacing_the_blob() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let other_commit = "abcdef0123456789012345678901234567890123";
        let first_repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let second_repo =
            Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = format!("{}.safetensors", "a".repeat(64));
        for commit in [COMMIT, other_commit] {
            let directory = snapshot(&cache, &first_repo, commit);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("model.safetensors"), fixture(&[&name])).unwrap();
        }
        let etag = "c".repeat(64);
        let resolve = format!("/owner/model/resolve/{COMMIT}/{name}");
        let mut routes = std::collections::HashMap::new();
        routes.insert(
            "/api/models/owner/model/revision/gb10".into(),
            (metadata_response(), Vec::new()),
        );
        routes.insert(
            resolve.clone(),
            (file_metadata(COMMIT, &etag, 4), file_body(&etag, b"data")),
        );
        routes.insert(
            "/api/models/owner/model/revision/gb202".into(),
            (revision_response(other_commit), Vec::new()),
        );
        routes.insert(
            format!("/owner/model/resolve/{other_commit}/{name}"),
            (
                file_metadata(other_commit, &etag, 4),
                file_body(&etag, b"data"),
            ),
        );
        let (endpoint, worker) = mock_hub_flexible(routes, 7);
        let api = test_api(&cache, endpoint);
        let (first, pinned) = fetch_primary(&api, &cache, &first_repo, None).unwrap();
        fetch_dependency(&api, &cache, &pinned, &first, &name).unwrap();
        let (second, pinned) = fetch_primary(&api, &cache, &second_repo, None).unwrap();
        fetch_dependency(&api, &cache, &pinned, &second, &name).unwrap();
        worker.join().unwrap();
        assert_eq!(
            second,
            snapshot(&cache, &second_repo, other_commit).join("model.safetensors")
        );
        let blob = cache.repo(first_repo.clone()).blob_path(&etag);
        assert_eq!(std::fs::read(&blob).unwrap(), b"data");
        #[cfg(unix)]
        {
            assert_same_file(&blob, &first.parent().unwrap().join(&name));
            assert_same_file(&blob, &second.parent().unwrap().join(&name));
        }
    }

    #[test]
    fn concurrent_fetches_transfer_the_payload_once() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        let lock = blob_lock(&blob).unwrap();
        let package = fixture(&[]);
        let resolve = format!("/owner/model/resolve/{COMMIT}/model.safetensors");
        let (ready, waiting) = std::sync::mpsc::channel();
        let (endpoint, worker) = mock_hub_steps(
            vec![
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (
                    resolve.clone(),
                    file_metadata(COMMIT, "fixture", package.len()),
                ),
                (resolve, file_response(&package)),
            ],
            move |step| {
                if step == 1 {
                    ready.send(()).unwrap();
                }
            },
        );
        let clients: Vec<_> = (0..2)
            .map(|_| {
                let cache = cache.clone();
                let repo = repo.clone();
                let endpoint = endpoint.clone();
                thread::spawn(move || {
                    fetch_file(
                        &test_api(&cache, endpoint),
                        &cache,
                        &repo,
                        "model.safetensors",
                    )
                    .map_err(|error| error.to_string())
                })
            })
            .collect();
        // Both callers have missed the snapshot and fetched its identity before either
        // can publish a blob. The waiting caller must reuse the completed transfer.
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(lock);
        for client in clients {
            let path = client.join().unwrap().unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), package);
            #[cfg(unix)]
            assert_same_file(&blob, &path);
        }
        worker.join().unwrap();
    }

    #[test]
    fn truncated_cached_blobs_are_rejected_without_modifying_them() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let blob = cache.repo(repo.clone()).blob_path("fixture");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"truncated").unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
            file_metadata(COMMIT, "fixture", 100),
        )]);
        let error = fetch_file(
            &test_api(&cache, endpoint),
            &cache,
            &repo,
            "model.safetensors",
        )
        .unwrap_err();
        worker.join().unwrap();
        assert!(
            error
                .to_string()
                .contains("invalid cached Hugging Face blob")
        );
        assert_eq!(std::fs::read(blob).unwrap(), b"truncated");
        assert!(!snapshot(&cache, &repo, COMMIT).exists());
    }

    #[test]
    fn cache_lookup_rejects_escaping_etags_and_unpinned_commits() {
        for (commit, etag) in [
            (COMMIT, "../outside"),
            (COMMIT, ""),
            ("other-commit", "fixture"),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let cache = Cache::new(temp.path().to_owned());
            let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
            let (endpoint, worker) = mock_hub(vec![(
                format!("/owner/model/resolve/{COMMIT}/model.safetensors"),
                file_metadata(commit, etag, 100),
            )]);
            let error = fetch_file(
                &test_api(&cache, endpoint),
                &cache,
                &repo,
                "model.safetensors",
            )
            .unwrap_err();
            worker.join().unwrap();
            assert!(
                error.to_string().contains("invalid file identity"),
                "{error}"
            );
            assert!(!snapshot(&cache, &repo, COMMIT).exists());
        }
    }
}
