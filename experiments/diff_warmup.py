def f(w):
    return w**3

def f_prime(w):
    return 3*(w**2)

def forward_diff(f, w, h):
    return (f(w + h) - f(w)) / h

def central_diff(f, w, h):
    return (f(w + h) - f(w - h)) / (2*h)

if __name__ == "__main__":
    w = 2.0
    true_value = f_prime(w)

    for h in [1e-2, 1e-3, 1e-5, 1e-15]:
        fd = forward_diff(f, w, h)
        cd = central_diff(f, w, h)
        print(f"h={h:.0e} | forward={fd:.10f} hata={abs(fd - true_value):.2e} | central={cd:.10f} hata={abs(cd - true_value):.2e}")