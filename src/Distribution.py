import os
import matplotlib.pyplot as plt


def get_distribution(data_path : str) -> list:
    """
    Takes a directory's path, gets the subdirectories name and counts their files.
    Since the subdirectories can belong to multiple plant categories (ex: apple, grape...) we also retrieve it.
    """
    distribution_list = []
    if data_path[-1] != '/':
        data_path += '/'
    for name in os.listdir(data_path):
        distribution_list.append({"category" : name.split('_', 1)[0],
                                  "name" : name,
                                  "size" : len(os.listdir(data_path + name))})
    return distribution_list

def display_distribution(dist_list : list) -> None:
    cat_list = []
    for d in dist_list: # cree liste des categories -> pourrait etre fusionné dans get_distribution
        if d["category"] not in cat_list:
            cat_list.append(d["category"])
    i = 0
    fig, ax = plt.subplots(len(cat_list), 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title("Distribution.py")
    for cat in cat_list: # cree graph en f des categories
        # passe autant qu'il y a de cate meme sur ce qui correspond pas -> 50% d'it en trop -> peut faire un {"cat" : id...}
        size_l = []
        name_l = []
        for sample in dist_list:
            if sample["category"] == cat:
                size_l.append(sample["size"])
                name_l.append(sample["name"])
        wedges, texts, autotexts = ax[i, 0].pie(size_l, labels=name_l, autopct='%1.1f%%')
        colors = [wedge.get_facecolor() for wedge in wedges]
        ax[i, 1].bar(name_l, size_l, color=colors)
        ax[i, 0].set_title(f"{cat}s :")
        i += 1
    plt.show()

l = get_distribution("/Users/theoke/Dev/leaffliction/images")
display_distribution(l)
