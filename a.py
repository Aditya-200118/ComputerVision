import matplotlib.pyplot as plt
import matplotlib.animation as animation

s = "zxyzxyz"
n = len(s)

fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(0, n)
ax.set_ylim(0, 1)
ax.axis("off")

# Draw characters
chars = []
for i, c in enumerate(s):
    chars.append(ax.text(i + 0.5, 0.5, c, ha="center", va="center", fontsize=20))

# Sliding window rectangle
window = plt.Rectangle((0, 0.3), 0, 0.4, color="green", alpha=0.3)
ax.add_patch(window)

info = ax.text(0.01, 0.85, "", transform=ax.transAxes, fontsize=12)

left = 0
right = 0
char_set = set()
max_len = 0

def update(frame):
    global left, right, max_len

    if right < n:
        if s[right] not in char_set:
            char_set.add(s[right])
            right += 1
            max_len = max(max_len, right - left)
        else:
            char_set.remove(s[left])
            left += 1

    # Update window position
    window.set_x(left)
    window.set_width(max(right - left, 0.01))

    # Reset colors
    for t in chars:
        t.set_color("black")

    # Highlight window characters
    for i in range(left, right):
        chars[i].set_color("green")

    info.set_text(
        f"Window: '{s[left:right]}'\nMax Length: {max_len}"
    )

    return chars + [window, info]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=30,
    interval=700,
    repeat=False
)

plt.show()
