def train(num):
    global steping
    for i in range(num):
        k = i^2
    steping += 1

if __name__ == '__main__':
    steping = 12
    train(1000)
    print(steping)