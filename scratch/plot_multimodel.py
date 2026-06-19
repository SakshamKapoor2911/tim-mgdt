import matplotlib.pyplot as plt
import numpy as np

# Data
t_values = [1, 2, 3]

mistral_ppl = [9.163, 9.121, 9.120]
llama_ppl = [12.265, 12.163, 12.107]
gemma_ppl = [22.037, 20.461, 19.493]

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Cross-Architecture Semantic Bleed: Boundary Thickness ($t$) vs. Healed PPL', fontsize=16, fontweight='bold', y=1.05)

# 1. Mistral
ax1.plot(t_values, mistral_ppl, marker='o', color='blue', linewidth=2.5, markersize=8)
ax1.set_title('Mistral-7B (MHA)\n"Hard Plateau"', fontsize=13, fontweight='bold', color='darkblue')
ax1.set_xlabel('Boundary Thickness ($t$)', fontsize=12)
ax1.set_ylabel('Healed C4 PPL', fontsize=12)
ax1.set_xticks([1, 2, 3])
ax1.grid(True, linestyle='--', alpha=0.6)

# Annotations for Mistral
for i, txt in enumerate(mistral_ppl):
    ax1.annotate(f'{txt:.3f}', (t_values[i], mistral_ppl[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# 2. Llama
ax2.plot(t_values, llama_ppl, marker='s', color='green', linewidth=2.5, markersize=8)
ax2.set_title('Meta-Llama-3-8B (GQA)\n"Hard Plateau"', fontsize=13, fontweight='bold', color='darkgreen')
ax2.set_xlabel('Boundary Thickness ($t$)', fontsize=12)
ax2.set_xticks([1, 2, 3])
ax2.grid(True, linestyle='--', alpha=0.6)

for i, txt in enumerate(llama_ppl):
    ax2.annotate(f'{txt:.3f}', (t_values[i], llama_ppl[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# 3. Gemma
ax3.plot(t_values, gemma_ppl, marker='^', color='red', linewidth=2.5, markersize=8)
ax3.set_title('Gemma-2-2B (SWA)\n"Monotonic Decay"', fontsize=13, fontweight='bold', color='darkred')
ax3.set_xlabel('Boundary Thickness ($t$)', fontsize=12)
ax3.set_xticks([1, 2, 3])
ax3.grid(True, linestyle='--', alpha=0.6)

for i, txt in enumerate(gemma_ppl):
    ax3.annotate(f'{txt:.3f}', (t_values[i], gemma_ppl[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/thickness_ablation_multimodel.png', dpi=300, bbox_inches='tight')
print("Saved paper/figures/thickness_ablation_multimodel.png")
