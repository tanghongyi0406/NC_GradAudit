# Publishing this package on GitHub

## 1. Review licenses and remove anything that cannot be redistributed

The code can be published directly. The three dataset archives require a
separate rights review. If image redistribution is not permitted, do not commit
the archives. Publish identifiers, permitted annotations, checksums, and an
approved data-preparation route instead.

## 2. Install Git and Git LFS

```bash
git --version
git lfs install
```

The included `.gitattributes` routes the compressed dataset archives
through Git LFS. Confirm this before the first commit:

```bash
git lfs track
git check-attr filter data_archives/*.tar.gz
```

## 3. Create the local repository

Run these commands inside the release directory:

```bash
git init
git branch -M main
git add .
git status
git diff --cached --stat
git commit -m "Release GradAudit pre-trained VLM reproduction code"
```

Inspect `git status` carefully. The extracted `data/`, downloaded `models/`,
and generated `results/` directories must not be staged.

## 4. Create and push the GitHub repository

Using the GitHub CLI:

```bash
gh auth login
gh repo create gradaudit-pretrained-vlms --public --source=. --remote=origin
git push -u origin main
```

Or create an empty repository in the GitHub web interface and run:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/gradaudit-pretrained-vlms.git
git push -u origin main
```

Large-file uploads may require a Git LFS data plan or an institutional release
host. Do not split archives merely to bypass platform or licensing controls.

## 5. Validate from a clean clone

```bash
git clone git@github.com:YOUR_ACCOUNT/gradaudit-pretrained-vlms.git clean-test
cd clean-test
git lfs pull
bash scripts/prepare_data.sh
```

Finally, run at least one complete model experiment from the clean clone and
compare its bootstrap output with the paper before making the repository public.
