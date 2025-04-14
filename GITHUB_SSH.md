Here's a quick tutorial for setting up a Git repository with SSH on different operating systems: Mac, Linux, and Windows.

### Prerequisites

- Install Git:
  - **Mac:** Use Homebrew: `brew install git`
  - **Linux:** Use a package manager (e.g., `sudo apt install git` on Ubuntu)
  - **Windows:** Install Git Bash from [git-scm.com](https://git-scm.com/)

### Steps

#### 1. Generate SSH Key

- **Mac/Linux:**
  ```bash
  ssh-keygen -t ed25519 -C "your_email@example.com"
  ```

- **Windows (Git Bash):**
  ```bash
  ssh-keygen -t ed25519 -C "your_email@example.com"
  ```

Press Enter to use the default location. When prompted, optionally enter a passphrase.

#### 2. Add SSH Key to SSH Agent

- **Mac/Linux:**
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```

- **Windows (Git Bash):**
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```

#### 3. Add SSH Key to GitHub

1. Copy the public key to your clipboard:
   - **Mac/Linux:**
     ```bash
     cat ~/.ssh/id_ed25519.pub | pbcopy
     ```
   - **Windows (Git Bash):**
     ```bash
     clip < ~/.ssh/id_ed25519.pub
     ```

2. Go to GitHub:
   - Navigate to **Settings** > **SSH and GPG keys** > **New SSH key**.
   - Paste the key and add a descriptive title.

#### 4. Clone a Repository Using SSH

- Clone a repository using SSH:
  ```bash
  git clone git@github.com:username/repository.git
  ```

#### 5. Set the Remote (if needed)

- Change an existing repository's remote URL to SSH:
  ```bash
  git remote set-url origin git@github.com:username/repository.git
  ```

#### 6. Verify SSH Connection

- Test the SSH connection:
  ```bash
  ssh -T git@github.com
  ```

This should confirm successful authentication. Now, you can use Git commands to interact with your repository over SSH!