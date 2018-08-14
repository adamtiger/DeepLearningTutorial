

def print_progress(status, max_iter, verbose):
    if (status + 1) % (max_iter/20) == 0 and verbose:
        print("Iterating: [%d%%]\r" %int((status+1)/max_iter * 100), end="")
        if int((status + 1)/max_iter) == 1:
            print("")

