from human_mouse import MouseController
import time

def move_mouse_naturally(target_x, target_y):
    """
    Moves the mouse to the specified coordinates with natural, human-like movement.

    Args:
        target_x (int): The target x-coordinate.
        target_y (int): The target y-coordinate.
    """
    mouse = MouseController()
    print(f"Moving mouse to ({target_x}, {target_y}) with natural movement...")
    mouse.move(target_x, target_y)
    print("Mouse movement complete.")

if __name__ == '__main__':
    # Example usage:
    # Move the mouse to a specific location on the screen.
    # Replace these coordinates with a desired target location.
    example_target_x = 500
    example_target_y = 500
    
    print("Starting example of natural mouse movement in 3 seconds...")
    time.sleep(3)
    
    move_mouse_naturally(example_target_x, example_target_y)
    
    print("
Moving to a random position in 3 seconds...")
    time.sleep(3)
    
    mouse = MouseController()
    mouse.move_random()
    print("Random movement complete.")
