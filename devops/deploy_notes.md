# Deploy Notes

A step-by-step checklist to get Paakzir live on the internet for free, written
for someone who has never deployed anything before. Do Part 1 (backend)
first — you'll need its URL for Part 2 (frontend).

---

## Part 1 — Backend on Render.com

1. Go to [render.com](https://render.com) and click **Get Started** (or
   **Sign In** if you already have an account). Sign up with your GitHub
   account — this makes the next steps much easier.
2. Once you're on the Render dashboard, click the **New +** button in the top
   right, then choose **Web Service** from the dropdown.
3. Render will ask you to connect a repository. Find and select
   `bluetreesandredskies/kangri-cancer-screening` from the list. (If you
   don't see it, click **Configure account** and grant Render access to that
   repo.)
4. On the setup form, fill in:
   - **Name**: `kangri-backend` (this becomes part of your live URL, so pick
     something recognizable).
   - **Region**: whichever is closest to you — it doesn't affect
     functionality.
   - **Branch**: `main` (or whichever branch you're deploying).
   - **Root Directory**: leave this **blank** — the build needs the whole
     repo, not just `backend/`.
   - **Runtime**: select **Docker**. Render should auto-detect this once it
     sees a Dockerfile.
   - **Dockerfile Path**: `backend/Dockerfile`.
5. Scroll down to **Instance Type** and select **Free**.
6. You don't need to add any environment variables for the backend — skip the
   "Advanced" section unless you added your own config.
7. Click the **Create Web Service** / **Deploy Web Service** button at the
   bottom.
8. Render will start building your Docker image — this can take several
   minutes the first time (it's installing PyTorch). Watch the **Logs** tab;
   you're looking for a line like `Uvicorn running on http://0.0.0.0:8000`.
9. Once the build finishes and the status badge at the top of the page turns
   green ("Live"), find your live URL at the top of the service page — it
   looks like `https://kangri-backend.onrender.com`. **Copy this URL**, you
   need it for Part 2.
10. Test it by opening `https://kangri-backend.onrender.com/health` in your
    browser — you should see `{"status":"ok"}`. (The free tier spins down
    after inactivity, so the very first request after a while may take
    10–30 seconds to wake back up — that's normal, not a bug.)

---

## Part 2 — Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) and click **Sign Up**, then choose
   **Continue with GitHub**.
2. From your Vercel dashboard, click **Add New...** → **Project**.
3. Find `bluetreesandredskies/kangri-cancer-screening` in the repository list
   and click **Import**. (If it's not listed, click **Adjust GitHub App
   Permissions** and grant access.)
4. On the **Configure Project** screen:
   - **Framework Preset**: Vercel should auto-detect **Vite** — leave it as
     is.
   - **Root Directory**: click **Edit** next to it and set it to `frontend`.
     This is important — without it, Vercel will try to build the whole repo
     instead of just the frontend.
5. Expand the **Environment Variables** section on the same screen (or find
   it under **Settings → Environment Variables** after the project is
   created). Add:
   - **Key**: `VITE_API_URL`
   - **Value**: the Render backend URL you copied in Part 1, step 9 (e.g.
     `https://kangri-backend.onrender.com`) — **no trailing slash**.
   - Leave it applied to all environments (Production, Preview, Development).
6. Click **Deploy**.
7. Wait for the build to finish (usually under a minute for a Vite app).
   Vercel shows a preview screenshot and a **Visit** button once it's done.
8. Click **Visit**, or copy the URL shown on the project's Overview page
   (something like `https://kangri-cancer-screening.vercel.app`) — this is
   your shareable live demo link.
9. Test the full flow: upload a sample lesion photo, click **Analyze**, and
   confirm a risk badge appears. If you instead see "Couldn't reach the
   server," double check the `VITE_API_URL` value from step 5 has no typo and
   matches your Render URL exactly, then redeploy (Vercel doesn't apply env
   variable changes to a deployment already in progress or already live —
   you need to trigger a new deploy, e.g. via **Deployments → ⋯ → Redeploy**).

---

## Quick troubleshooting

- **Backend build fails on Render**: open the **Logs** tab and check for a
  missing dependency or an out-of-memory error (PyTorch + the model
  checkpoint can be heavy for the free tier's build resources).
- **Frontend shows a blank page**: open the browser console — a common cause
  is the Root Directory not being set to `frontend` in step 4 above.
- **CORS errors in the browser console**: shouldn't happen since the backend
  allows all origins, but if it does, confirm the Render service is actually
  "Live" and not still building.
