

def split_and_subset(array, grid_size=8):
    # Resize the array if not divisible by grid size
    y, x, c = array.shape
    if y % grid_size != 0 or x % grid_size != 0:
        array = resize_to_nearest_divisible(array, grid_size)
        y, x, c = array.shape  # Update dimensions after resizing
    
    # Calculate tile dimensions
    tile_height = y // grid_size
    tile_width = x // grid_size
    
    # Split into tiles
    tiles = []
    for i in range(0, y, tile_height):
        for j in range(0, x, tile_width):
            tiles.append(array[i:i+tile_height, j:j+tile_width, :])
    
    # Reshape tiles into a grid
    tiles = np.array(tiles).reshape(grid_size, grid_size, tile_height, tile_width, c)
    
    # Subset: Remove edge tiles (first and last row/column)
    inner_tiles = tiles[1:-1, 1:-1]  # Exclude edges
    
    # Stack inner tiles along a single axis
    stacked_tiles = inner_tiles.reshape(-1, tile_height, tile_width, c)
    return stacked_tiles
