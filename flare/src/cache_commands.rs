//! Prepared-model cache commands: list, inspect, select, remove, and gc.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use flare_helios::package::cache::{self, Root};

use crate::cache_support::resolve_model;
use crate::inspect::{Canvas, bytes, text, thousands};
use serde_json::Value;

use crate::cli::{
    CacheArgs, CacheCommand, CacheGc, CacheList, CacheRemove, CacheSpec,
};

pub fn run(args: CacheArgs) -> Result<(), Box<dyn std::error::Error>> {
    let root = || -> Result<Root, Box<dyn std::error::Error>> {
        Ok(Root::open(
            crate::cache_support::model_root(args.model_cache.as_deref())
                .map_err(|error| error.to_string())?,
        )?)
    };
    match args.command {
        CacheCommand::List(list) => list_entries(&root()?, list, args.plain),
        CacheCommand::Inspect(mut inspect) => {
            inspect.plain |= args.plain;
            crate::inspect::run(inspect)
        }
        CacheCommand::Download(download) => crate::hub::run_download(download),
        CacheCommand::Select(spec) => select(&root()?, spec, args.plain),
        CacheCommand::Remove(remove) => remove_entry(&root()?, remove, args.plain),
        CacheCommand::Gc(gc) => collect(&root()?, gc, args.plain),
    }
}

fn split(spec: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let (model, version) = spec.rsplit_once('@').ok_or("expected MODEL@VERSION")?;
    if model.is_empty() || version.is_empty() {
        return Err("expected MODEL@VERSION".into());
    }
    Ok((model.to_owned(), version.to_owned()))
}

/// Unique-inode accounting for one entry tree.
///
/// Logical bytes are the sizes recorded in the manifest and in the files. Physical bytes are the
/// allocated blocks, so sparse files and hard links are not over-counted as disk usage. On
/// platforms without block accounting the physical figures are reported as unavailable.
#[derive(Default)]
struct Usage {
    logical_bytes: u64,
    physical_bytes: Option<u64>,
    shared_logical_bytes: u64,
    shared_physical_bytes: Option<u64>,
    single_physical_bytes: Option<u64>,
    files: usize,
    shared_objects: usize,
}

#[cfg(unix)]
fn usage(directory: &Path) -> Result<Usage, Box<dyn std::error::Error>> {
    use std::os::unix::fs::MetadataExt;
    let mut seen: BTreeMap<(u64, u64), (u64, u64, u64)> = BTreeMap::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if !metadata.is_file() {
                continue;
            }
            seen.insert(
                (metadata.dev(), metadata.ino()),
                (metadata.len(), metadata.blocks() * 512, metadata.nlink()),
            );
        }
    }
    let mut usage = Usage {
        physical_bytes: Some(0),
        shared_physical_bytes: Some(0),
        single_physical_bytes: Some(0),
        ..Default::default()
    };
    for (logical, physical, links) in seen.values() {
        usage.logical_bytes += logical;
        usage.physical_bytes = usage.physical_bytes.map(|bytes| bytes + physical);
        usage.files += 1;
        if *links > 1 {
            usage.shared_logical_bytes += logical;
            usage.shared_physical_bytes = usage.shared_physical_bytes.map(|bytes| bytes + physical);
            usage.shared_objects += 1;
        } else {
            usage.single_physical_bytes = usage.single_physical_bytes.map(|bytes| bytes + physical);
        }
    }
    Ok(usage)
}

#[cfg(not(unix))]
fn usage(directory: &Path) -> Result<Usage, Box<dyn std::error::Error>> {
    let mut seen: BTreeMap<PathBuf, u64> = BTreeMap::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if metadata.is_file() {
                seen.insert(entry.path(), metadata.len());
            }
        }
    }
    let mut usage = Usage::default();
    for (_, logical) in seen {
        usage.logical_bytes += logical;
        usage.files += 1;
    }
    Ok(usage)
}


fn list_entries(root: &Root, list: CacheList, plain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let models = match &list.model {
        Some(model) => vec![resolve_model(root, model).map_err(|error| error.to_string())?],
        None => root
            .models()?
            .into_iter()
            .map(|key| {
                let identity = cache::model_identity(&key).unwrap_or_else(|_| key.clone());
                (key, identity)
            })
            .collect(),
    };
    let mut documents = Vec::new();
    for (model, identity) in models {
        let default = root.ref_entry(&model)?;
        let versions = root.versions(&model)?;
        let mut entries = Vec::new();
        for version in versions {
            let selected = default.as_deref() == Some(version.as_str());
            let report = (|| -> Result<(bool, flare_helios::package::cache::EntrySummary, Usage), Box<dyn std::error::Error>> {
                let leased = root.is_leased(&model, &version)?;
                let entry = root.entry(&model, &version)?;
                let summary = root.summarize_entry(&entry)?;
                let usage = usage(entry.directory())?;
                Ok((leased, summary, usage))
            })();
            let (leased, summary, usage) = match report {
                Ok(values) => values,
                Err(error) => {
                    // One unreadable entry must not hide the rest of the cache.
                    entries.push(serde_json::json!({
                        "model": identity,
                        "model_key": model,
                        "version": version,
                        "default": selected,
                        "error": error.to_string(),
                    }));
                    continue;
                }
            };
            // Physical reclamation is only attributable when this entry owns the only link to
            // its bytes and nothing leases it. Otherwise report why it is unavailable.
            let physical = usage.physical_bytes;
            let (reclaimable, reclaimable_note) = if leased {
                (None, Some("leased by a running process"))
            } else if selected {
                (None, Some("selected default"))
            } else if physical.is_none() {
                (
                    None,
                    Some("allocated-block accounting is unavailable on this platform"),
                )
            } else if usage.shared_objects > 0 {
                (
                    usage.single_physical_bytes,
                    Some(
                        "allocated bytes of single-linked inodes only; shared or externally hardlinked objects are excluded",
                    ),
                )
            } else {
                (physical, None)
            };
            entries.push(serde_json::json!({
                "model": identity,
                "model_key": model,
                "version": version,
                "default": selected,
                "leased": leased,
                "metadata": "unverified",
                "form": summary.form,
                "format_version": summary.format_version,
                "model_identity": summary.model_identity,
                "hardware_identity": summary.hardware_identity,
                "payloads": summary.payloads,
                "logical_bytes": summary.logical_bytes,
                "disk_logical_bytes": usage.logical_bytes,
                "disk_physical_bytes": physical,
                "shared_logical_bytes": usage.shared_logical_bytes,
                "shared_physical_bytes": usage.shared_physical_bytes,
                "files": usage.files,
                "shared_objects": usage.shared_objects,
                "reclaimable_bytes": reclaimable,
                "reclaimable_note": reclaimable_note,
                "deployments": summary.deployments.iter().map(|deployment| serde_json::json!({
                    "id": deployment.id,
                    "tensor_parallel": deployment.tensor_parallel,
                    "ready": deployment.ready,
                    "pages": deployment.pages,
                    "state_rows": deployment.state_rows,
                    "max_sequence_tokens": deployment.max_sequence_tokens,
                    "prefill_chunk_size": deployment.prefill_chunk_size,
                })).collect::<Vec<_>>(),
            }));
        }
        documents.push(serde_json::json!({
            "model": identity,
            "model_key": model,
            "default": default,
            "versions": entries,
        }));
    }
    let document = serde_json::json!({
        "cache": root.path().display().to_string(),
        "models": documents,
    });
    if list.json {
        println!("{}", serde_json::to_string_pretty(&document)?);
    } else {
        print!("{}", render_list(&document, Canvas::terminal(plain)));
    }
    Ok(())
}

fn entry_result(title: &str, model: &str, version: &str, plain: bool) {
    let mut output = Canvas::terminal(plain);
    output.heading(title);
    output.row("Model", &text(&Value::String(model.into())));
    output.row("Version", &text(&Value::String(version.into())));
    print!("{}", output.finish());
}

fn amount(value: &Value) -> String {
    value.as_u64().map_or_else(|| "unavailable".into(), bytes)
}

fn count(value: &Value) -> String {
    value.as_u64().map_or_else(|| "—".into(), thousands)
}

fn render_list(document: &Value, mut output: Canvas) -> String {
    let models = document["models"].as_array().unwrap();
    let versions: usize = models.iter().map(|model| model["versions"].as_array().unwrap().len()).sum();
    output.heading("Local models");
    output.row("Storage", &text(&document["cache"]));
    if versions == 0 {
        output.note("No local model versions found.");
        return output.finish();
    }
    output.row("Available", &format!("{} model{} · {} version{}",
        models.len(), if models.len() == 1 { "" } else { "s" },
        versions, if versions == 1 { "" } else { "s" }));
    for model in models {
        output.heading(&text(&model["model"]));
        for entry in model["versions"].as_array().unwrap() {
            let mut status = Vec::new();
            if entry["default"] == true { status.push("DEFAULT"); }
            if entry["leased"] == true { status.push("IN USE"); }
            if entry.get("error").is_some() { status.push("UNREADABLE"); }
            let version = text(&entry["version"]);
            output.row("Version", &if status.is_empty() { version } else {
                format!("{version}  [{}]", status.join(" · "))
            });
            if let Some(error) = entry.get("error") {
                output.row("Error", &text(error));
                output.note("");
                continue;
            }
            output.row("Format", &format!("{} v{}", text(&entry["form"]), text(&entry["format_version"])));
            output.row("Disk used", &amount(&entry["disk_physical_bytes"]));
            output.row("Shared files", &format!("{} · {} hardlinked objects",
                amount(&entry["shared_logical_bytes"]), count(&entry["shared_objects"])));
            output.row("Payload / file size", &format!("{} / {}",
                if entry["form"] == "legacy" { "unavailable".into() } else { amount(&entry["logical_bytes"]) },
                amount(&entry["disk_logical_bytes"])));
            let reclamation = if entry["leased"] == true {
                "unavailable · in use by a running process".into()
            } else if entry["default"] == true {
                "unavailable · selected default".into()
            } else if entry["reclaimable_bytes"].is_null() {
                "unavailable · filesystem accounting unsupported".into()
            } else {
                format!("{}{}", amount(&entry["reclaimable_bytes"]),
                    if entry["shared_objects"].as_u64().unwrap_or(0) > 0 { " · excludes hardlinked files" } else { "" })
            };
            output.row("Reclaimable", &reclamation);
            for deployment in entry["deployments"].as_array().unwrap() {
                output.row("Deployment", &format!("{} · TP {} · {}", text(&deployment["id"]),
                    count(&deployment["tensor_parallel"]),
                    if deployment["ready"] == true { "ready" } else { "metadata only" }));
                output.row("Context / prefill", &format!("{} / {} tokens",
                    count(&deployment["max_sequence_tokens"]), count(&deployment["prefill_chunk_size"])));
                output.row("Pages / state rows", &format!("{} / {}", count(&deployment["pages"]), count(&deployment["state_rows"])));
            }
            output.note("");
        }
    }
    output.note("Capacity metadata is unverified. Use flare models inspect MODEL@VERSION for verified details.");
    output.finish()
}

fn select(root: &Root, spec: CacheSpec, plain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, version) = split(&spec.spec)?;
    let (model, identity) = resolve_model(root, &spec).map_err(|error| error.to_string())?;
    root.publish(&model, &version)?;
    entry_result("Default model selected", &identity, &version, plain);
    Ok(())
}

fn remove_entry(root: &Root, remove: CacheRemove, plain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, version) = split(&remove.spec)?;
    let (model, identity) = resolve_model(root, &spec).map_err(|error| error.to_string())?;
    root.remove(&model, &version, remove.clear_default)?;
    entry_result("Model version removed", &identity, &version, plain);
    Ok(())
}

fn collect(root: &Root, gc: CacheGc, plain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let collected = root.collect()?;
    if gc.json {
        println!(
            "{}",
            serde_json::json!({
                "staging_trees": collected.staging,
                "payload_objects": collected.objects,
                "legacy_containers": collected.legacy,
            })
        );
    } else {
        let mut output = Canvas::terminal(plain);
        output.heading("Unused model storage freed");
        output.row("Temporary directories", &thousands(collected.staging as u64));
        output.row("Model files", &thousands(collected.objects as u64));
        output.row("Legacy model files", &thousands(collected.legacy as u64));
        if collected.staging == 0 && collected.objects == 0 && collected.legacy == 0 {
            output.note("Nothing to reclaim.");
        }
        print!("{}", output.finish());
    }
    Ok(())
}

/// Unused on platforms without hard-linked cache entries; keeps the module portable.
#[allow(dead_code)]
fn _path_type(_: PathBuf) -> Option<PathBuf> {
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn usage_counts_allocated_blocks_and_shared_inodes() {
        let directory = std::env::temp_dir().join(format!(
            "flare-cache-usage-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let sparse = directory.join("sparse.bin");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&sparse).unwrap();
            file.set_len(8 << 20).unwrap();
            file.write_all(&[1u8; 4096]).unwrap();
        }
        std::fs::hard_link(&sparse, directory.join("shared.bin")).unwrap();
        let usage = usage(&directory).unwrap();
        assert_eq!(usage.files, 1, "one inode counts once");
        assert_eq!(usage.logical_bytes, 8 << 20);
        let physical = usage.physical_bytes.unwrap();
        assert!(
            physical < usage.logical_bytes,
            "sparse data must not count as disk usage"
        );
        assert_eq!(usage.shared_objects, 1);
        assert_eq!(usage.shared_logical_bytes, 8 << 20);
        assert_eq!(usage.single_physical_bytes, Some(0));
        let _ = std::fs::remove_dir_all(directory);
    }
}
