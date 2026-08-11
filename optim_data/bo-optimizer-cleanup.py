if __name__ == "__main__":
    listA = []  # params to filter

    listB = []  # targets to filter

    # new bounds to check
    inds_to_keep = []
    for i in range(len(listA)):
        if listA[i][0] >= 2004:
            inds_to_keep.append(i)
    print(inds_to_keep, len(listA), len(listB))

    print("PARAMS:", [listA[i] for i in inds_to_keep])
    print()
    print("TARGETS", [listB[i] for i in inds_to_keep])
    print()
