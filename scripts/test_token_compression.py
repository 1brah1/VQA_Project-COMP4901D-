import torch

from src.vision.token_compression import compress_27x27_tokens


def test_shapes() -> None:
    x = torch.randn(2, 729, 16)
    for t in [729, 243, 81, 27, 9]:
        y = compress_27x27_tokens(x, target_tokens=t)
        assert y.shape == (2, t, 16)


def test_constant_preserved() -> None:
    x = torch.full((1, 729, 8), 3.14)
    for t in [243, 81, 27, 9]:
        y = compress_27x27_tokens(x, target_tokens=t)
        assert torch.allclose(y, torch.full((1, t, 8), 3.14))


if __name__ == "__main__":
    test_shapes()
    test_constant_preserved()
    print("ok")
