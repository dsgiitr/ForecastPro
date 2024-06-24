import pandas as pd
import matplotlib.pyplot as plt

#
# A function to plot the training history
# If you do not need to plot all the loss and accuracy,
# pass the list of the desired items in dict_keys
#
def PlotHistory(history,
                title = None,
                xlabel = None,
                ylabel = None,
                xlim = None,
                ylim = None,
                grid = True,
                figsize = (8,5),
                dict_keys = None):
    print(type(dict_keys))
    # Ensure history exists and is of type dict
    assert(history is not None and type(history) is dict)

    if dict_keys is not None and type(dict_keys) is list:
        # if we do not want to plot everything
        pd.DataFrame({key: history[key] for key in dict_keys}).plot(figsize=figsize)
    else:
        pd.DataFrame(history).plot(figsize=figsize)
    plt.grid(grid)
    
    if xlim is not None:
        plt.gca().set_xlim(xlim)

    if ylim is not None:
        plt.gca().set_ylim(ylim)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
