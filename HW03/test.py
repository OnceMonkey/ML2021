import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paragrams!')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--sum', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args)
    print(args.sum(args.integers))
    print(args.lr)

