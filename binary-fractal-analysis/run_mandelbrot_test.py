#!/usr/bin/env python3
"""Quick test of prime Mandelbrot/Mandelbulb analysis."""

import math
import random

def sieve_primes(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def first_n_primes(n):
    ln_n = math.log(n)
    limit = int(n * (ln_n + math.log(ln_n) + 2))
    primes = sieve_primes(limit)
    return primes[:n]

def mandelbrot_escape(c_real, c_imag, max_iter=500):
    zr, zi = 0.0, 0.0
    for n in range(max_iter):
        zr2, zi2 = zr*zr, zi*zi
        if zr2 + zi2 > 4.0:
            return n
        zi = 2*zr*zi + c_imag
        zr = zr2 - zi2 + c_real
    return max_iter

def mandelbulb_escape(cx, cy, cz, power=8, max_iter=100):
    x, y, z = 0.0, 0.0, 0.0
    for n in range(max_iter):
        r = math.sqrt(x*x + y*y + z*z)
        if r > 2.0:
            return n
        if r < 1e-10:
            x, y, z = cx, cy, cz
            continue
        theta = math.atan2(y, x)
        phi = math.acos(z / r)
        r_n = r ** power
        theta_n = theta * power
        phi_n = phi * power
        x = r_n * math.sin(phi_n) * math.cos(theta_n) + cx
        y = r_n * math.sin(phi_n) * math.sin(theta_n) + cy
        z = r_n * math.cos(phi_n) + cz
    return max_iter

# Generate primes
print('Generating 20,000 primes...')
primes = first_n_primes(20000)
gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
print(f'Primes: 2 to {primes[-1]:,}')
print(f'Gap range: [{min(gaps)}, {max(gaps)}]')
print()

# 2D Mandelbrot: c = scale * (g_n + i*g_{n+1})
print('='*60)
print('2D MANDELBROT: c = 0.1 * (g_n + i*g_{n+1})')
print('='*60)

scale = 0.1
real_esc = []
for i in range(len(gaps) - 1):
    esc = mandelbrot_escape(gaps[i]*scale, gaps[i+1]*scale, 500)
    real_esc.append(esc)

real_mean = sum(real_esc) / len(real_esc)
real_in = sum(1 for e in real_esc if e >= 500)
print(f'Real: mean escape = {real_mean:.2f}, in M-set = {real_in}')

# Shuffled
shuf_means = []
shuf_ins = []
for trial in range(5):
    shuf = gaps.copy()
    random.shuffle(shuf)
    shuf_esc = [mandelbrot_escape(shuf[i]*scale, shuf[i+1]*scale, 500) for i in range(len(shuf)-1)]
    shuf_means.append(sum(shuf_esc)/len(shuf_esc))
    shuf_ins.append(sum(1 for e in shuf_esc if e >= 500))

print(f'Shuffled: mean escape = {sum(shuf_means)/5:.2f}, in M-set = {sum(shuf_ins)/5:.1f}')
ratio_mean = real_mean / (sum(shuf_means)/5)
ratio_in = real_in / (sum(shuf_ins)/5) if sum(shuf_ins) > 0 else float('inf')
print(f'Ratio (mean): {ratio_mean:.3f}')
print(f'Ratio (in-set): {ratio_in:.3f}')

if abs(ratio_mean - 1.0) > 0.05:
    print(f'*** DIFFERENCE: {abs(ratio_mean-1)*100:.1f}% ***')
print()

# 3D Mandelbulb: c = scale * (g_n, g_{n+1}, g_{n+2})
print('='*60)
print('3D MANDELBULB: c = 0.1 * (g_n, g_{n+1}, g_{n+2})')
print('='*60)

real_esc_3d = []
for i in range(len(gaps) - 2):
    esc = mandelbulb_escape(gaps[i]*scale, gaps[i+1]*scale, gaps[i+2]*scale, 8, 100)
    real_esc_3d.append(esc)

real_mean_3d = sum(real_esc_3d) / len(real_esc_3d)
real_in_3d = sum(1 for e in real_esc_3d if e >= 100)
print(f'Real: mean escape = {real_mean_3d:.2f}, in Bulb = {real_in_3d}')

# Shuffled 3D
shuf_means_3d = []
shuf_ins_3d = []
for trial in range(5):
    shuf = gaps.copy()
    random.shuffle(shuf)
    shuf_esc = [mandelbulb_escape(shuf[i]*scale, shuf[i+1]*scale, shuf[i+2]*scale, 8, 100) for i in range(len(shuf)-2)]
    shuf_means_3d.append(sum(shuf_esc)/len(shuf_esc))
    shuf_ins_3d.append(sum(1 for e in shuf_esc if e >= 100))

print(f'Shuffled: mean escape = {sum(shuf_means_3d)/5:.2f}, in Bulb = {sum(shuf_ins_3d)/5:.1f}')
ratio_mean_3d = real_mean_3d / (sum(shuf_means_3d)/5)
ratio_in_3d = real_in_3d / (sum(shuf_ins_3d)/5) if sum(shuf_ins_3d) > 0 else float('inf')
print(f'Ratio (mean): {ratio_mean_3d:.3f}')
print(f'Ratio (in-set): {ratio_in_3d:.3f}')

if abs(ratio_mean_3d - 1.0) > 0.05:
    print(f'*** DIFFERENCE: {abs(ratio_mean_3d-1)*100:.1f}% ***')
print()

# What triplets land inside?
print('='*60)
print('INSIDE THE MANDELBULB')
print('='*60)
inside = [(gaps[i], gaps[i+1], gaps[i+2]) for i in range(len(gaps)-2)
          if mandelbulb_escape(gaps[i]*scale, gaps[i+1]*scale, gaps[i+2]*scale, 8, 100) >= 100]
print(f'Triplets inside: {len(inside)}')
if inside:
    from collections import Counter
    c = Counter(inside)
    print('Most common:')
    for t, cnt in c.most_common(10):
        print(f'  {t}: {cnt}')

# Compare different scales
print()
print('='*60)
print('SCALE SENSITIVITY')
print('='*60)
for s in [0.05, 0.08, 0.1, 0.12, 0.15]:
    real_e = [mandelbulb_escape(gaps[i]*s, gaps[i+1]*s, gaps[i+2]*s, 8, 100) for i in range(min(5000, len(gaps)-2))]
    shuf = gaps[:5002].copy()
    random.shuffle(shuf)
    shuf_e = [mandelbulb_escape(shuf[i]*s, shuf[i+1]*s, shuf[i+2]*s, 8, 100) for i in range(min(5000, len(shuf)-2))]

    r_mean = sum(real_e)/len(real_e)
    s_mean = sum(shuf_e)/len(shuf_e)
    r_in = sum(1 for e in real_e if e >= 100)
    s_in = sum(1 for e in shuf_e if e >= 100)

    print(f'scale={s:.2f}: Real mean={r_mean:.2f} in={r_in}, Shuf mean={s_mean:.2f} in={s_in}, ratio={r_mean/s_mean:.3f}')
