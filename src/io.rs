//! Bounded local image and JSON files. Output creation is exclusive.
use crate::{Error, GrayImage, Result, error::PartialPublish};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

const MAX_JSON_BYTES: u64 = 128 * 1024 * 1024;
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_JSON_BYTES {
        return Err(Error::InvalidInput(
            "JSON file exceeds 128 MiB limit".into(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_JSON_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(Error::InvalidInput("JSON file exceeds size limit".into()));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut transaction = OutputTransaction::new([path])?;
    transaction.stage_json(path, value)?;
    transaction.commit()
}

pub fn load_image(path: &Path) -> Result<GrayImage> {
    let mut reader =
        image::ImageReader::new(BufReader::new(File::open(path)?)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode()?.to_luma8();
    GrayImage::new(
        image.width() as usize,
        image.height() as usize,
        image.into_raw(),
    )
}

pub fn save_image(path: &Path, image: &GrayImage) -> Result<()> {
    let mut transaction = OutputTransaction::new([path])?;
    transaction.stage_image(path, image)?;
    transaction.commit()
}

/// Files staged in existing directories and published only after every writer succeeds.
///
/// Each final path is exclusive: an existing file or a path collision is rejected, never replaced.
/// Multi-file publication is not atomic; a later failure may leave earlier paths published.
pub struct OutputTransaction {
    entries: Vec<OutputEntry>,
}

struct OutputEntry {
    target: PathBuf,
    staged: Option<NamedTempFile>,
}

impl OutputTransaction {
    pub fn new<I, P>(paths: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut entries = Vec::new();
        for path in paths {
            let target = canonical_target(path.as_ref())?;
            if entries
                .iter()
                .any(|entry: &OutputEntry| entry.target == target)
            {
                return Err(Error::InvalidInput(format!(
                    "duplicate output path: {}",
                    path.as_ref().display()
                )));
            }
            match fs::symlink_metadata(&target) {
                Ok(_) => {
                    return Err(Error::InvalidInput(format!(
                        "output already exists: {}",
                        target.display()
                    )));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            entries.push(OutputEntry {
                target,
                staged: None,
            });
        }
        if entries.is_empty() {
            return Err(Error::InvalidInput("output transaction is empty".into()));
        }
        Ok(Self { entries })
    }

    pub fn stage_json<T: Serialize>(&mut self, path: &Path, value: &T) -> Result<()> {
        let index = self.entry_index(path)?;
        self.stage(index, |file| {
            serde_json::to_writer(&mut *file, value)?;
            file.write_all(b"\n")?;
            Ok(())
        })
    }

    pub fn stage_image(&mut self, path: &Path, image: &GrayImage) -> Result<()> {
        image.validate()?;
        let index = self.entry_index(path)?;
        let format = image::ImageFormat::from_path(&self.entries[index].target)?;
        let pixels =
            image::GrayImage::from_raw(image.width as u32, image.height as u32, image.data.clone())
                .ok_or_else(|| Error::InvalidInput("invalid image dimensions".into()))?;
        self.stage(index, |file| {
            image::DynamicImage::ImageLuma8(pixels).write_to(file, format)?;
            Ok(())
        })
    }

    pub fn commit(mut self) -> Result<()> {
        if self.entries.iter().any(|entry| entry.staged.is_none()) {
            return Err(Error::InvalidInput(
                "every output must be staged before commit".into(),
            ));
        }
        let mut published = Vec::with_capacity(self.entries.len());
        for entry in &mut self.entries {
            let Some(staged) = entry.staged.take() else {
                if !published.is_empty() {
                    return Err(PartialPublish::new(
                        published,
                        std::io::Error::other("output was not staged"),
                    )
                    .into());
                }
                return Err(Error::InvalidInput(
                    "every output must be staged before commit".into(),
                ));
            };
            match staged.persist_noclobber(&entry.target) {
                Ok(_) => {
                    published.push(entry.target.clone());
                    #[cfg(test)]
                    tests::after_publish(&entry.target);
                }
                Err(error) => {
                    entry.staged = Some(error.file);
                    if !published.is_empty() {
                        return Err(PartialPublish::new(published, error.error).into());
                    }
                    return Err(error.error.into());
                }
            }
        }
        Ok(())
    }

    fn entry_index(&self, path: &Path) -> Result<usize> {
        let target = canonical_target(path)?;
        self.entries
            .iter()
            .position(|entry| entry.target == target)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "path was not declared for this transaction: {}",
                    path.display()
                ))
            })
    }

    fn stage<F>(&mut self, index: usize, write: F) -> Result<()>
    where
        F: FnOnce(&mut File) -> Result<()>,
    {
        if self.entries[index].staged.is_some() {
            return Err(Error::InvalidInput(format!(
                "output was already staged: {}",
                self.entries[index].target.display()
            )));
        }
        let parent = self.entries[index]
            .target
            .parent()
            .ok_or_else(|| Error::InvalidInput("output target has no parent".into()))?;
        let mut staged = NamedTempFile::new_in(parent)?;
        write(staged.as_file_mut())?;
        staged.as_file_mut().flush()?;
        staged.as_file().sync_all()?;
        self.entries[index].staged = Some(staged);
        Ok(())
    }
}

fn canonical_target(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            Error::InvalidInput(format!("output path has no filename: {}", path.display()))
        })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = parent.canonicalize()?;
    if !parent.is_dir() {
        return Err(Error::InvalidInput(format!(
            "output parent is not a directory: {}",
            parent.display()
        )));
    }
    Ok(parent.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, process::Command};

    type PublishHook = Box<dyn FnMut(&Path)>;
    thread_local! {
        static HOOK: RefCell<Option<PublishHook>> = RefCell::new(None);
    }

    pub(super) fn after_publish(path: &Path) {
        HOOK.with(|hook| {
            if let Some(callback) = hook.borrow_mut().as_mut() {
                callback(path);
            }
        });
    }

    #[test]
    fn replacement_actor() {
        let Some(raw) = std::env::var_os("IRISTRUST_SYNTHETIC_ACTOR_DIR") else {
            return;
        };
        let root = PathBuf::from(raw).canonicalize().unwrap();
        let allowed = std::env::temp_dir().canonicalize().unwrap();
        assert_eq!(root.parent(), Some(allowed.as_path()));
        assert_eq!(
            std::fs::read(root.join("owned-fixture-marker")).unwrap(),
            b"SYNTHETIC_OWNED_FIXTURE"
        );
        let first = root.join("first.json");
        let saved = root.join("saved-original.json");
        assert!(first.is_file() && !saved.exists());
        std::fs::rename(&first, &saved).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&first)
            .unwrap();
        file.write_all(b"SYNTHETIC_REPLACEMENT_SENTINEL").unwrap();
    }

    #[test]
    fn concurrent_replacement_is_never_removed_after_later_failure() {
        let temp_root = std::env::temp_dir().canonicalize().unwrap();
        let dir = tempfile::tempdir_in(&temp_root).unwrap();
        let root = dir.path().canonicalize().unwrap();
        assert_eq!(root.parent(), Some(temp_root.as_path()));
        let mut marker = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("owned-fixture-marker"))
            .unwrap();
        marker.write_all(b"SYNTHETIC_OWNED_FIXTURE").unwrap();
        drop(marker);
        let first = root.join("first.json");
        let second = root.join("second.json");
        let mut transaction = OutputTransaction::new([&first, &second]).unwrap();
        transaction
            .stage_json(&first, &"synthetic original")
            .unwrap();
        transaction
            .stage_json(&second, &"synthetic second")
            .unwrap();
        let mut conflict = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&second)
            .unwrap();
        conflict.write_all(b"SYNTHETIC_CONFLICT").unwrap();
        drop(conflict);

        let actor_root = root.clone();
        HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |path| {
                assert_eq!(path, actor_root.join("first.json"));
                let result = Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "io::tests::replacement_actor"])
                    .env("IRISTRUST_SYNTHETIC_ACTOR_DIR", &actor_root)
                    .output()
                    .unwrap();
                assert!(result.status.success());
            }));
        });
        let result = transaction.commit();
        HOOK.with(|hook| *hook.borrow_mut() = None);
        let error = result.unwrap_err();
        let Error::PartialPublish(partial) = error else {
            panic!("expected partial publication");
        };
        assert_eq!(partial.published_paths(), &[first.clone()]);
        assert_eq!(partial.cause().kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(&first).unwrap(),
            b"SYNTHETIC_REPLACEMENT_SENTINEL"
        );
        assert_eq!(std::fs::read(&second).unwrap(), b"SYNTHETIC_CONFLICT");
        assert!(root.join("saved-original.json").is_file());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 4);
    }
}
