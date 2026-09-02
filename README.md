# voiceagent

Development happens on volatile VM (`ssh box` → `/workspace`). This GitHub repo is the persistent backup — commit & push regularly to save progress.

## Workflow

### On `box` (VM)
```bash
# first time
git clone https://github.com/mamajjou/voiceagent.git ~/voiceagent
cd ~/voiceagent
# or if /workspace is the repo:
# git clone https://github.com/mamajjou/voiceagent.git /tmp/voiceagent && cp -a /tmp/voiceagent/.git /workspace/.git && etc

# daily work
git add -A
git commit -m "feat: ..."
git push origin main
```

### From host (this machine)
```bash
ssh box
cd /workspace  # or ~/voiceagent
git pull
# work, then push
```

### Save script (on box)
```bash
./scripts/save.sh "your commit message"
```

## Initial setup on box
Box has no `gh` CLI — uses HTTPS with token. Token is available via `gh auth token` on the host. To set up:

```bash
# on host
gh auth token | ssh box "cat > /tmp/token && git config --global credential.helper store && echo \"https://oauth2:\$(cat /tmp/token)@github.com\" > ~/.git-credentials && rm /tmp/token"
ssh box "git clone https://oauth2:\$(cat ~/.git-credentials | cut -d: -f3 | cut -d@ -f1)@github.com/mamajjou/voiceagent.git ~/voiceagent 2>&1 || echo clone failed, trying existing"
```

