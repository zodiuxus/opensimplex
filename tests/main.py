from opensimplex import OpenSimplex
import torch, time

def opensimplex_test(device: str):
    generator = torch.Generator(device=device)
    start = time.time()
    os = OpenSimplex(generator=generator)
    end = time.time()
    return os.noise2(10,10), device, end-start

print(opensimplex_test('cuda'))
print('')
print(opensimplex_test('cpu'))