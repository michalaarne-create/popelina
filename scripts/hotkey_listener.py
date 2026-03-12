import keyboard
import pyautogui
import random
import math
from natural_mouse import move_mouse_naturally

def get_random_point_in_radius(radius):
    """
    Generates a random point within a given radius from the current mouse position.

    Args:
        radius (int): The maximum distance from the current position.

    Returns:
        tuple: A tuple containing the new x and y coordinates.
    """
    current_x, current_y = pyautogui.position()
    
    # Generate a random angle and distance
    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.uniform(0, 1)) # This ensures uniform distribution within the circle
    
    # Calculate the new point
    new_x = int(current_x + r * math.cos(angle))
    new_y = int(current_y + r * math.sin(angle))
    
    # Ensure the new point is within the screen bounds
    width, height = pyautogui.size()
    new_x = max(0, min(width - 1, new_x))
    new_y = max(0, min(height - 1, new_y))
    
    return new_x, new_y

def on_hotkey_press():
    """
    This function is called when the hotkey is pressed.
    It triggers the natural mouse movement to a random point.
    """
    print("Hotkey '?' pressed. Moving mouse to a random nearby location.")
    target_x, target_y = get_random_point_in_radius(500)
    move_mouse_naturally(target_x, target_y)

def main():
    """
    Main function to set up and run the hotkey listener.
    """
    hotkey = "?"
    print(f"Listening for hotkey '{hotkey}'. Press '{hotkey}' to trigger mouse movement.")
    print("Press '}' to stop the listener.")

    # Register the hotkey
    keyboard.add_hotkey(hotkey, on_hotkey_press)

    # Keep the script running to listen for the hotkey
    # The script will exit when the '}' key is pressed.
    keyboard.wait('}')
    print("Listener stopped.")

if __name__ == "__main__":
    main()
