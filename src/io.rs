//! Bounded local image and JSON files. Output creation is exclusive.
use crate::{Error, GrayImage, Result};
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

/// Files staged in one existing directory and published only after every writer succeeds.
///
/// Each final path is exclusive: an existing file or a path collision is rejected, never replaced.
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
                for target in published {
                    let _ = fs::remove_file(target);
                }
                return Err(Error::InvalidInput(
                    "every output must be staged before commit".into(),
                ));
            };
            match staged.persist_noclobber(&entry.target) {
                Ok(_) => published.push(entry.target.clone()),
                Err(error) => {
                    entry.staged = Some(error.file);
                    for target in published {
                        let _ = fs::remove_file(target);
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
