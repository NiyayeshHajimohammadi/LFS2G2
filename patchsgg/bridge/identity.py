from patchsgg.bridge.base import Bridge


class IdentityBridge(Bridge):
    """No modality-gap handling (relies on the shared Talk2DINO space alone)."""

    def forward(self, cond, *, training, modality):
        return cond
