# UrzaLLM – embedding job

Thanks for running this! It turns about 2,500 short Czech texts (articles, book
passages and video descriptions) into vectors with the open model
[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Everything runs locally on your
computer; nothing is uploaded anywhere. The only download is the model itself
(about 2.3 GB, from Hugging Face) plus the Python packages.

## What you need

- **Python 3.10–3.13** ([python.org](https://www.python.org/downloads/); on Windows tick
  "Add python.exe to PATH" during install).
- About **6 GB of free disk** (packages + model) and **8 GB of RAM**.
- A GPU is optional. Rough times: NVIDIA GPU ~2–5 min, Apple Silicon Mac ~10–20 min,
  CPU only ~30–90 min.

## Run it

**Windows:** double-click `run.bat` (or run `run.bat` in a terminal in this folder).

**Mac / Linux:** open a terminal in this folder and run:

```bash
bash run.sh
```

The script creates a private Python environment in `.venv` inside this folder,
installs the packages, checks the setup with a quick self-test, embeds everything and
shows progress with an estimate of the time left.

When it finishes it prints **`Send this file back: …results_<date>.zip`**.
Please send that zip back. That's all.

### Windows with an NVIDIA GPU

On Windows, the default PyTorch install uses only the CPU. To use the GPU, install the
CUDA build **before** the first run:

1. Run `run.bat --dry-run` once (it sets up `.venv` and checks the data; it doesn't need the model).
2. Go to <https://pytorch.org/get-started/locally/>, choose Stable / Windows / Pip / Python /
   the CUDA version, and copy the command it shows.
3. In this folder run `.venv\Scripts\activate`, then
   `pip uninstall -y torch`, then paste the command from step 2.
4. Run `run.bat` again. It should print `Device: cuda`.

## If something goes wrong

- **It stopped or crashed halfway:** just run it again. Finished batches are kept in
  `output/partial/` and skipped.
- **Out of memory:** run with a smaller batch, e.g. `run.bat --batch-size 4`
  (or `bash run.sh --batch-size 4`).
- **Force the CPU:** add `--device cpu`.
- Anything else: send back the messages from the terminal.

`run.bat --dry-run` / `bash run.sh --dry-run` checks the data and the output files with
random vectors, without downloading the model. Its zip is named `DRYRUN_not_real_…`;
please don't send that one.

## What's in this folder

| File | |
|---|---|
| `data/chunks.parquet` | The texts to embed (`chunk_id`, `source`, `embed_text`). |
| `embed.py` | The script. |
| `run.bat`, `run.sh` | Set up the environment and run the script. |
| `requirements.txt` | Python packages (PyTorch is installed by the run scripts). |

The result zip contains `embeddings.npy` (one 1,024-number vector per text),
`chunk_ids.txt` (which text each row belongs to) and `manifest.json` (model version,
device, timings and checksums).
