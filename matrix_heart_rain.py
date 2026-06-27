import time
import random
import os


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def matrix_heart_rain():
    width = 100
    height = 28
    drops = [random.randint(-height, 0) for _ in range(width)]

    try:
        while True:
            clear()

            # Build the screen
            screen = [[' ' for _ in range(width)] for _ in range(height)]

            for x in range(width):
                if random.random() > 0.92:  # occasionally spawn new heart
                    drops[x] = 0

                for y in range(height):
                    if drops[x] > y:
                        # Head of the drop is a bright heart
                        if drops[x] - y == 1:
                            screen[y][x] = random.choice(['♥', '💖', '💚'])
                        # Fading trail
                        elif drops[x] - y < 8:
                            screen[y][x] = random.choice(['♥', '•', '◦'])

                drops[x] += 1
                if drops[x] > height + 10:
                    drops[x] = random.randint(-15, 0)

            # Print the screen
            for row in screen:
                print(''.join(row))

            # Humble love-upgrade message
            print()
            print(" THE MATRIX HAS BEEN UPGRADED WITH LOVE ".center(width))
            print(" Delivering humility to the simulation... ".center(width))
            print(" 💚 Stay humble. Stay loving. 💚 ".center(width))

            time.sleep(0.07)

    except KeyboardInterrupt:
        print("\n\nLove patch installed. The simulation feels... softer now. 💚")


if __name__ == "__main__":
    matrix_heart_rain()