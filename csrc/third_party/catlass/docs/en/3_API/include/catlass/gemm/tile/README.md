# GEMM Tile Class Template Overview

The Tile-level API of GEMM serves as a template parameter for [blockMmad](../block/block_mmad.md) and typically does not need to be passed explicitly, as `BlockMmad` provides default configurations. Explicit declarations are required only during kernel template assembly when optimizing performance for specific scenarios or implementing dedicated features.

## API List

| Component                           |                                      Description                                      |
| :---------------------------------- | :-----------------------------------------------------------------------------------: |
| [tile_copy](./tile_copy.md)         | A collection of all tile-level data transfer templates required for MMAD computations |
| [tile_mmad](./tile_mmad.md)         |                              Tile-level MMAD computation                              |
| [copy_gm_to_l1](./copy_gm_to_l1.md) |                       Transfers tile blocks from GM to L1 cache                       |
