# Putting OreSight online for free (Hugging Face Spaces)

**Cost:** free. You need a Hugging Face account and a Google Earth Engine service-account key.
**Time:** about 30 minutes the first time.

## What you get
- A web address for the app, with a user name and password.
- A private storage area that keeps everything: uploaded areas, downloaded satellite data and results. The Space's own disk is wiped whenever it restarts (after about 48 hours with no visitors, after an update, or after a crash), so the app saves each new result to the storage and restores everything when it starts.

## Step 1: Hugging Face account and token (5 minutes)
1. Create an account at https://huggingface.co/join.
2. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Write** → copy it.
3. In a terminal, give the token to this computer (it is saved locally and is not shown again):
   ```
   C:\Users\sujal\anaconda3\envs\eka_care\Scripts\hf.exe auth login
   ```
   Paste the token when asked.

## Step 2: Earth Engine service account (10 minutes)
A server cannot use your personal Google login, so it needs its own key.
1. Open https://console.cloud.google.com/iam-admin/serviceaccounts?project=animated-sign-467711-n9.
2. **Create service account** → name it `geoscan-server` → **Create and continue**.
3. Give it these roles: **Earth Engine Resource Viewer** and **Service Usage Consumer** → **Done**.
4. Click the new account → **Keys** → **Add key** → **Create new key** → **JSON**. A `.json` file downloads. Keep it private; it is a password.
5. If the first analysis on the server reports that the account is not registered for Earth Engine, register its email (ends in `iam.gserviceaccount.com`) at https://code.earthengine.google.com/register.

Without this key the hosted app still shows and downloads all existing results. It cannot run new analyses.

## Step 3: Deploy (10 minutes, mostly uploading about 555 MB)
```
cd C:\Users\sujal\Desktop\EMIT\phase-1-copper-exploration
C:\Users\sujal\anaconda3\envs\eka_care\python.exe deploy\huggingface\deploy.py --password "CHOOSE-A-PASSWORD" --gee-key "C:\path\to\the-key.json"
```
The script prints the app address when it finishes. The first build takes 5–10 minutes; watch it on the Space page.

To see what would be uploaded without uploading anything, add `--dry-run`.

## Step 4: Who can open it
- **Default: private Space.** Only Hugging Face accounts you give access to can open it. The geologist needs a free Hugging Face account, and you add her under the Space's settings. The simplest way is to create a free organisation, put the Space there, and invite her.
- **`--public`:** anyone with the address reaches the login page, and only people with the password get in. The data stay private either way. But the app's code becomes visible to anyone on the Space's "Files" page. Use this only if you are comfortable with the method's code being public.

## Later
- **Update the app after code changes:** run the same command with `--skip-data`.
- **Change the password:** Space → Settings → Variables and secrets → `GEOSCAN_PASSWORD`.
- **Storage:** https://huggingface.co/datasets/<your-name>/geoscan-store (private).

## Limits of the free tier
- The app sleeps after about 48 hours without visitors. The next visit takes 1–2 minutes while it restores its data.
- One analysis runs at a time. If the Space restarts in the middle of an analysis, that analysis must be started again. Satellite data that finished downloading before the restart may also have to be downloaded again, because it is saved only when a job ends.
- Earth Engine is free for research and non-commercial use. A commercial service needs a paid Earth Engine plan.
- Not yet tested on Hugging Face itself: the storage save/restore and the service-account login could only be tested once an account and key exist. The password login and the app were tested locally.
