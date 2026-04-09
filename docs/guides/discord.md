# Discord Voice Conversion

Let others hear your converted voice on Discord calls.

## How It Works

```
Your mic → mlx-vc (real-time VC) → BlackHole (virtual device) → Discord reads as mic
                                  ↘ Your headphones (monitor, so you hear yourself too)
```

**BlackHole** is a free virtual audio device for macOS. It creates a "fake microphone" — anything your program outputs to it, Discord can read as mic input.

## Step 1: Install BlackHole

```bash
brew install blackhole-2ch
```

Or download from [existential.audio/blackhole](https://existential.audio/blackhole/).

After installing, **restart your Mac**.

## Step 2: Verify Installation

```bash
python -m mlx_vc.demo.realtime_vc --reference speaker.wav --list-devices
```

You should see BlackHole in the list:

```
  [10] BlackHole 2ch (IN/OUT, 44100Hz) [BlackHole]
```

## Step 3: Run with `--discord`

```bash
python -m mlx_vc.demo.realtime_vc \
  --reference /path/to/target_speaker.wav \
  --discord
```

This automatically:

1. Detects BlackHole and outputs to it
2. Also outputs to your headphones so you can hear yourself (monitor)

## Step 4: Configure Discord

1. Open **Discord** → **Settings** (gear icon)
2. Go to **Voice & Video**
3. **Input Device** → select **BlackHole 2ch**
4. Test: speak and check if the input level bar moves

## Options

```bash
# Discord mode + specific input mic
python -m mlx_vc.demo.realtime_vc \
  --reference speaker.wav \
  --discord \
  --input-device 2

# Discord mode + specific monitor headphone device
python -m mlx_vc.demo.realtime_vc \
  --reference speaker.wav \
  --discord \
  --monitor 1

# Lower latency (fewer diffusion steps)
python -m mlx_vc.demo.realtime_vc \
  --reference speaker.wav \
  --discord \
  --diffusion-steps 5
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| BlackHole not in device list | Restart Mac after installing |
| Discord doesn't show BlackHole | Quit and reopen Discord |
| No audio in Discord | Check System Settings → Privacy & Security → Microphone (allow Discord) |
| You can't hear yourself | Add `--monitor <device_id>` for your headphones |
| High latency | Use `--diffusion-steps 5 --block-time 0.3` |
