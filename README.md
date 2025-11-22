# Pipe resonance estimation from videos

## Usage

Manage dependencies:

```bash
make create-env # create conda env called pipe_resonance
make update-env # update conda env called pipe_resonance if already exists
conda activate pipe_resonance # activate environment
```

Usage

```bash
python main.py "air_set/WhatsApp Video 2025-04-19 at 20.04.04.mp4" --debug-video debug.mp4 --plot
python main.py "air_set/WhatsApp Video 2025-04-19 at 20.06.10.mp4" --debug-video debug.mp4 --plot
python main.py "air_set/IMG_3128.MOV" --debug-video debug.mp4 --plot
python main.py "air_set/IMG_3129.MOV" --debug-video debug.mp4 --plot
```

Convert output video to whatsapp compatible

```bash
ffmpeg -i debug.mp4 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -crf 23 -movflags +faststart debug_wapp_comatible.mp4
```