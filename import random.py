import random
n=random.randint(1,10)
while True:
    g=int(input("Guess (1-10):"))
    if g==n:
        print("correct!");break
    print("try again")