# mnist_vae

# From Autoencoder to Variational Autoencoder: An Intuitive Journey

This project explains **Variational Autoencoders (VAEs)** not through equations or probability theory — but through **engineering intuition**.

We start with a simple **autoencoder** and try to generate new digits by sampling points in its latent space.  
When that doesn’t work, we improve the design step by step, introducing new constraints each time.  
Each step leads us closer to the principles behind the VAE.

---

## 🧠 The Idea

If similar digits are mapped close together in the autoencoder’s latent space,  
then sampling random points nearby *should* produce new digits.  

It sounds simple — but in practice, it doesn’t work.  
The generated digits look noisy or meaningless.  

So, we begin refining the design.  
Each improvement is motivated by an observation, not a formula.

---

## 🚶 The Journey — One Model at a Time

### 1️⃣ Autoencoder
- Learns to **compress** and **reconstruct** digits.
- Latent space clusters similar digits together.
- But when we **sample random points** from this space, the generated digits look messy.
- **Observation:** The latent space is unstructured — the model never learned what random regions should look like.

---

### 2️⃣ VAE Version 1 — Constrain the Latent Space
- We add a constraint to keep latent vectors **within a compact region**.
- The idea: prevent the latent space from drifting all over, so random samples fall in a meaningful zone.
- **Result:** Sampling improves slightly — we start to see digit-like shapes, but still not convincing.

---

### 3️⃣ VAE Version 2 — Encourage Smoothness
- We notice that nearby points in latent space can still produce very different digits.
- So we add a constraint: **neighboring points should decode to similar outputs.**
- This makes the latent space **smooth and continuous.**
- **Result:** Sampling produces more coherent digits, but the space still lacks flexibility.

---

### 4️⃣ VAE Version 3 — Learn Neighborhood Size and Shape
- Different digits may need different kinds of neighborhoods.
- So we let the model **learn its own neighborhood size and shape** for each sample.
- **Result:** Sampling from the latent space now produces realistic, diverse digits.

---

## 🎯 The Outcome

By the end of this process, we’ve arrived at the essence of a **Variational Autoencoder** —  
not by deriving equations, but by **engineering intuition**:

- We want the latent space to be compact (VAE prior).  
- We want nearby points to decode to similar outputs (continuity).  
- We want the model to learn how uncertainty varies across samples (variance).  

Each step mirrors the motivations behind the VAE’s formal loss function —  
but we got there by observing behavior, not by starting from math.

---

## 💻 Running the Demo

```bash
# Clone the repo
git clone https://github.com/mahesh-godavarti/mnist_vae.git
cd mnist_vae

# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the script
python mnist_vae.py
