import tkinter as tk

# List of available DPS characters in Overwatch
dps_list = [
    {"name": "Ashe", "hit_scan": True, "mobility": False, "high_ground_access": True, "is_sniper": True, "is_tracking": False, "projectile": False, "counters_snipers": False, "counters_pharah": True},
    {"name": "Bastion", "hit_scan": True, "mobility": False, "high_ground_access": False, "is_sniper": False, "is_tracking": True, "projectile": True, "counters_snipers": False, "counters_pharah": True},
    {"name": "Echo", "hit_scan": True, "mobility": True, "high_ground_access": True, "is_sniper": False, "is_tracking": True, "projectile": True, "counters_snipers": False, "counters_pharah": True},
    {"name": "Genji", "hit_scan": False, "mobility": True, "high_ground_access": True, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": True, "counters_pharah": False},
    {"name": "Hanzo", "hit_scan": False, "mobility": False, "high_ground_access": True, "is_sniper": True, "is_tracking": False, "projectile": True, "counters_snipers": True, "counters_pharah": False}, 
    {"name": "Junkrat", "hit_scan": False, "mobility": False, "high_ground_access": True, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": False, "counters_pharah": False},
    {"name": "McCree", "hit_scan": True, "mobility": False, "high_ground_access": False, "is_sniper": False, "is_tracking": False, "projectile": False, "counters_snipers": False, "counters_pharah": True},
    {"name": "Mei", "hit_scan": False, "mobility": False, "high_ground_access": False, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": False, "counters_pharah": False},
    {"name": "Pharah", "hit_scan": False, "mobility": True, "high_ground_access": True, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": False, "counters_pharah": False},
    {"name": "Reaper", "hit_scan": False, "mobility": False, "high_ground_access": True, "is_sniper": False, "is_tracking": True, "projectile": False, "counters_snipers": False, "counters_pharah": False},
    {"name": "Soldier: 76", "hit_scan": True, "mobility": True, "high_ground_access": False, "is_sniper": False, "is_tracking": True, "projectile": True, "counters_snipers": False, "counters_pharah": True},
    {"name": "Soujorn", "hit_scan": True, "mobility": True, "high_ground_access": True, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": False, "counters_pharah": True},
    {"name": "Sombra", "hit_scan": True, "mobility": True, "high_ground_access": True, "is_sniper": False, "is_tracking": True, "projectile": False, "counters_snipers": True, "counters_pharah": False},
    {"name": "Symmetra", "hit_scan": False, "mobility": False, "high_ground_access": True, "is_sniper": False, "is_tracking": True, "projectile": True, "counters_snipers": False, "counters_pharah": False},
    {"name": "Torbjörn", "hit_scan": False, "mobility": False, "high_ground_access": False, "is_sniper": False, "is_tracking": False, "projectile": True, "counters_snipers": False, "counters_pharah": False},
    {"name": "Tracer", "hit_scan": False, "mobility": True, "high_ground_access": False, "is_sniper": False, "is_tracking": True, "projectile": False, "counters_snipers": True, "counters_pharah": False},
    {"name": "Widowmaker", "hit_scan": True, "mobility": False, "high_ground_access": True, "is_sniper": True, "is_tracking": False, "projectile": False, "counters_snipers": True, "counters_pharah": True},
]

# Initialize a list to store the selected DPS characters
selected_dps = []


# Create a Tkinter window
window = tk.Tk()

# Create a label widget to display error messages
error_label = tk.Label(window, text="")
error_label.pack()


def check_dps(selected_dps):

    # Check if counters pharah
    has_pharah_counter = False
    for hero in selected_dps:
        if hero["counters_pharah"]:
            has_pharah_counter = True
            break
    if not has_pharah_counter:
        error_label.config(text="Your hero pool must include at least one hero that counters Pharah.", foreground="red")

    # Check if you have a mix of tracking and non-tracking heroes
    has_tracking = False
    has_non_tracking = False
    for hero in selected_dps:
        if hero["is_tracking"]:
            has_tracking = True
        elif hero["is_tracking"] == False:
            has_non_tracking = True
    if has_tracking and has_non_tracking:
        error_label.config(text="Your hero pool should include either all tracking heroes or all non-tracking heroes.", foreground="red")

    # Check if at least one hero in the pool has hit scan
    has_hit_scan = False
    for hero in selected_dps:
        if hero["hit_scan"]:
            has_hit_scan = True
            break
    if not has_hit_scan:
        error_label.config(text="Your hero pool must include at least one hero with hit scan.", foreground="red")

    has_sniper = False
    for hero in selected_dps:
        if hero["is_sniper"] or hero["counters_snipers"]:
            has_sniper = True
            break
    if not has_sniper:
        error_label.config(text="Your hero pool must include at least one sniper or counter.", foreground="red")

    # Check if at least one hero in the pool has high mobility
    has_high_mobility = False
    for hero in selected_dps:
        if hero["mobility"]:
            has_high_mobility = True
            break
    if not has_high_mobility:
        error_label.config(text="Your hero pool must include at least one hero with high mobility.", foreground="red")

    # Check if at least one hero in the pool has high ground access
    has_high_ground_access = False
    for hero in selected_dps:
        if hero["high_ground_access"]:
            has_high_ground_access = True
            break
    if not has_high_ground_access:
        error_label.config(text="Your hero pool must include at least one hero with easy high ground access.", foreground="red")

def add_dps():
    # Get the currently selected DPS character from the dropdown menu
    selected = next(hero for hero in dps_list if hero['name'] == var.get())
    # Add the selected DPS character to the list if it hasn't already been selected
    if selected not in selected_dps:
        selected_dps.append(selected)
        print(f"{selected['name']} has been added to your pool of DPS characters.")
        
        # Clear the error message if it was previously displayed
        error_label.config(text="")
        
        # Check if the hero pool meets the specified criteria
        check_dps(selected_dps)
        # Update the label widget with the current pool of selected DPS characters
        pool_label.config(text="Your current pool of DPS characters:\n" + "\n".join(hero["name"] for hero in selected_dps))


def clear_pool():
    # Clear the list of selected DPS characters
    selected_dps.clear()
    # Update the label widget with the current pool of selected DPS characters
    pool_label.config(text="Your current pool of DPS characters:\n" + "\n".join(hero["name"] for hero in selected_dps))
    # Clear the error message if it was previously displayed
    error_label.config(text="")

# Set the window title
window.title("Overwatch DPS Selector")

# Set the window size
window.geometry("500x500")

# Create a dropdown menu with the available DPS characters
var = tk.StringVar(window)
var.set(dps_list[0]['name'])
dropdown = tk.OptionMenu(window, var, *[hero['name'] for hero in dps_list])
dropdown.pack()

# Create a button to add the selected DPS character to the pool
add_button = tk.Button(window, text="Add to pool", command=add_dps)
add_button.pack()

# Create a button to add the selected DPS character to the pool
add_button = tk.Button(window, text="Clear pool", command=clear_pool)
add_button.pack()

# Create a label widget to display the current pool of selected DPS characters
pool_label = tk.Label(window, text="Your current pool of DPS characters:\n")
pool_label.pack()

# Start the Tkinter event loop
window.mainloop()
