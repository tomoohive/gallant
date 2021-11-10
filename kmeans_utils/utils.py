import random

def rand_ints_nodu(a, b, k):
    ns = []
    while len(ns) < k:
        n = random.randint(a, b)
        if not n in ns:
            ns.append(n)
    return ns

def rand_color_map(a, b, k):
    R = rand_ints_nodu(a, b, k)
    G = rand_ints_nodu(a, b, k)
    B = rand_ints_nodu(a, b, k)
    RGB = []
    for r,g,b in zip(R,G,B):
        RGB.append([r,g,b])
    return RGB