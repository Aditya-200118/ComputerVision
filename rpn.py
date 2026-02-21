import matplotlib.pyplot as plt
import matplotlib.animation as animation

tokens = ["1","2","+","3","*","4","-"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 5)
ax.set_ylim(0, 6)
ax.axis("off")

stack = []
step = 0

token_text = ax.text(0.5, 5.5, "", fontsize=14)
op_text = ax.text(0.5, 5.0, "", fontsize=12)

stack_texts = []

def draw_stack():
    global stack_texts
    for t in stack_texts:
        t.remove()
    stack_texts = []

    for i, val in enumerate(stack):
        txt = ax.text(
            2.5, i + 0.5, str(val),
            ha="center", va="center",
            fontsize=14,
            bbox=dict(boxstyle="round", fc="lightblue")
        )
        stack_texts.append(txt)

def update(frame):
    global step

    if step >= len(tokens):
        token_text.set_text(f"Result = {stack[-1]}")
        op_text.set_text("")
        return stack_texts + [token_text, op_text]

    c = tokens[step]
    token_text.set_text(f"Token: {c}")

    if c == "+":
        a, b = stack.pop(), stack.pop()
        stack.append(b + a)
        op_text.set_text(f"Compute: {b} + {a}")
    elif c == "-":
        a, b = stack.pop(), stack.pop()
        stack.append(b - a)
        op_text.set_text(f"Compute: {b} - {a}")
    elif c == "*":
        a, b = stack.pop(), stack.pop()
        stack.append(b * a)
        op_text.set_text(f"Compute: {b} * {a}")
    elif c == "/":
        a, b = stack.pop(), stack.pop()
        stack.append(int(float(b) / a))
        op_text.set_text(f"Compute: {b} / {a}")
    else:
        stack.append(int(c))
        op_text.set_text("Push to stack")

    draw_stack()
    step += 1

    return stack_texts + [token_text, op_text]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(tokens) + 2,
    interval=1500,
    repeat=False
)

plt.show()
