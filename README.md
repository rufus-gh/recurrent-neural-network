# Recurrent Neural Network

A character-level recurrent neural network, written from scratch in numpy, that learnt to write Shakespeare.

You type the start of some text and the network carries it on one character at a time, picking each next character based on everything it has seen so far.

```
> ROMEO:
I thy rime sat swuto sale, and mery,
ter s hit mawe as yoal ne thar her gend min we lovnmepI
I hou miSind nopelt,
```

It isn't going to win any awards, but it learnt names, line breaks, punctuation and roughly what English words look like, starting from nothing but the raw text.

## Try it

**[Download it](https://github.com/rufus-gh/recurrent-neural-network/releases/latest)** for macOS, Linux or Windows. No Python needed, the trained model is inside the binary.

On macOS and Linux:

```bash
chmod +x rnn-*
xattr -d com.apple.quarantine rnn-* 2>/dev/null   # macOS only, the binary is unsigned
./rnn-macos-arm64
```

On Windows run the `.exe` from a terminal. SmartScreen will warn you it's unsigned.

Or run it from source. A trained model is included, so you don't need to train anything first:

```bash
git clone https://github.com/rufus-gh/recurrent-neural-network && cd recurrent-neural-network/
pip install -r requirements.txt
python main.py # or python3 main.py depending on your version
```

Type something at the `>` prompt and press enter. An empty line lets it start from scratch. Ctrl+C to quit.

## What it does

- Generates text character by character in the style of Shakespeare
- Two stacked RNN layers of 512 hidden units each
- Forward pass, backpropagation through time and Adagrad all written by hand, no ML libraries
- Trained on ~1MB of Shakespeare (`data/input.txt`)
- Can train your own model on any text file

## Training your own

```bash
python main.py --train
```

Runs 20,000 iterations (a few minutes), saves the model to `my-model.npz`, then lets you type prompts to it. The included model was trained for about 1.4 million iterations, so expect yours to be a lot rougher unless you leave it going.

| Flag | What it does | Default |
| --- | --- | --- |
| `--train` | Train a new model instead of just loading one | off |
| `-i`, `--iterations` | Number of training iterations | `20000` |
| `-d`, `--data` | Text file to train on | `data/input.txt` |
| `-o`, `--output` | Output filename, no extension | `my-model` |
| `-m`, `--model` | Model to load when not training | `model_checkpoint.npz` |
| `-n`, `--length` | How many characters to generate per prompt | `500` |
| `--graph` | Plot the training loss at the end | off |

```bash
python main.py --train -i 100000 --graph -o my-model
python main.py -m my-model.npz -n 1000
```

## How it works

The text is split into chunks of 25 characters. Each character is turned into a one-hot vector and fed through two tanh RNN layers, where each layer's hidden state is carried over from the previous character. The top layer's output goes through a softmax to give a probability for every possible next character, and the loss is how unlikely it thought the real next character was.

Backpropagation through time goes back over the 25 steps to get the gradients, which are clipped to ±5 so they don't explode, then Adagrad updates the weights. The learning rate slowly decays, and the hidden state is reset every 100 chunks.

To generate, the network reads your prompt to build up its hidden state, then samples a character from its output, feeds that back in, and repeats.

## Files

- `main.py` - training and generating
- `model_checkpoint.npz` - the pretrained model
- `data/input.txt` - the Shakespeare it was trained on
- `train.ipynb` - the notebook I trained it in

Training saves to `my-model.npz` by default, so the pretrained one stays put.

## Releases

All three binaries are built by [.github/workflows/release.yml](.github/workflows/release.yml) when a `v*` tag is pushed. PyInstaller can't cross-compile, so each one builds on its own runner, generates some text to check it actually works, and then gets attached to the release:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

To build one locally instead:

```bash
pip install pyinstaller
pyinstaller --onefile --name rnn --add-data "model_checkpoint.npz:." \
  --add-data "data/input.txt:data" --exclude-module matplotlib \
  --runtime-hook packaging/runtime_hook.py main.py
```

Lands in `dist/`, about 20MB, nearly all of it numpy. On Windows the `--add-data` separator is `;` instead of `:`.

The runtime hook is needed because `main.py` loads the model and data by relative path, and in a bundled binary they get unpacked to a temp folder rather than the one you run from. The hook points the working directory at it.

## Credits

Made by [Rufus Gordon-Heywood](https://github.com/rufus-gh) for Hack Club Stardance. Uses numpy and matplotlib, everything else is written from scratch.
