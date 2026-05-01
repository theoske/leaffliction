import os
import matplotlib.pyplot as plt
import argparse


def is_dir(path: str):
    if not os.path.isdir(path):
        print(f"ERROR : folder \'{path}\' doesn't exist.")
        return False
    return True


def get_distribution(data_path: str) -> dict:
    """
    Takes a directory's path, gets the subdirectories name and counts their
    files.
    Since the subdirectories can belong to multiple plant categories
    (ex: apple, grape...) we also retrieve it.
    cat dict-> dirslists -> dirdict
    """
    if not is_dir(data_path):
        exit(-1)
    distrib_d = {}
    if data_path[-1] != '/':
        data_path += '/'
    for name in os.listdir(data_path):
        if name[0] == '.':
            print(f"WARNING: skipped folder \'{name}\'")
            continue
        category = name.split('_', 1)[0]
        try:
            size = len(os.listdir(data_path + name))
        except NotADirectoryError:
            print(f"WARNING: \'{name}\' is not a directory. Skipped it.")
            continue
        if category not in distrib_d:
            distrib_d[category] = []
        distrib_d[category].append({"category": category,
                                    "name": name,
                                    "size": size})
    return distrib_d


def display_distribution(distribution: dict) -> None:
    if len(distribution) < 2:
        print("ERROR: Not enough data for piechart, need 2.")
        exit(1)
    i = 0
    fig, ax = plt.subplots(len(distribution), 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title("Distribution.py")
    for category in distribution:
        size_l = []
        name_l = []
        for category_dict in distribution[category]:
            size_l.append(category_dict["size"])
            name_l.append(category_dict["name"])
        wedges, texts, autotexts = ax[i, 0].pie(size_l, labels=name_l,
                                                autopct='%1.1f%%')
        colors = [wedge.get_facecolor() for wedge in wedges]
        ax[i, 1].bar(name_l, size_l, color=colors)
        ax[i, 0].set_title(f"{category} :")
        i += 1
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", nargs="?", help="Base folder path",
                        default="../images")
    args = parser.parse_args()
    distrib = get_distribution(args.folder_path)
    if len(distrib) < 1:
        print(f"WARNING: no data was found in \'{args.folder_path}\'")
        exit(-1)
    display_distribution(distrib)


if __name__ == "__main__":
    main()
