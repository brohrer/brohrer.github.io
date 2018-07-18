import numpy as np
import numpy.matlib as mat
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d


def main():
    """
    Model a 2-variable function and show how a decision tree might
    partition it.
    Visualize in 3D.
    """
    month_min = 0
    month_max = 12
    age_min = 5
    age_max = 75
    time_min = 5.5
    time_max = 8.5

    data_spacing = 5
    n_pts = 300

    ages = np.arange(age_min, age_max + 1, data_spacing)
    months = np.arange(month_min, month_max + 1)
    ages_m = mat.repmat(ages, months.size, 1).T
    months_m = mat.repmat(months, ages.size, 1)
    times = np.zeros((ages.size, months.size))

    ages_hi = np.linspace(age_min, age_max, num=n_pts)
    months_hi = np.linspace(month_min, month_max, num=n_pts)
    ages_hi_m = mat.repmat(ages_hi, months_hi.size, 1).T
    months_hi_m = mat.repmat(months_hi, ages_hi.size, 1)
    model = 6.5 * np.ones((ages_hi.size, months_hi.size))

    def wakeup_time(age, month):
        """
        Generate wakeup time data, based on a model and some noise.
        """
        time = (
            # Age trend
            (5.8 + 2.5 * np.exp(-age / 20)

            # Seasonal variation multiplier
            * (1 + .3 * np.cos(2 * np.pi * month / 12)))

            # Some random variation
            * (1 + .05 * np.random.random_sample()))
        return time

    for i_age, age in enumerate(ages):
        for i_month, month in enumerate(months):
            times[i_age, i_month] = wakeup_time(age, month)

    figsize = (13, 5)
    fig1 = plt.figure(0, figsize=figsize)
    ax11 = fig1.add_subplot(121, projection='3d')
    ax12 = fig1.add_subplot(122)

    fig2 = plt.figure(1, figsize=figsize)
    ax21 = fig2.add_subplot(121, projection='3d')
    ax22 = fig2.add_subplot(122)

    p1 = ax12.pcolor(
        ages_m, months_m, times,
        cmap=plt.cm.viridis,
        vmax=time_max,
        vmin=time_min,
    )
    cb1 = fig1.colorbar(p1)
    p2 = ax22.pcolor(
        ages_hi_m, months_hi_m, model,
        cmap=plt.cm.viridis,
        vmax=time_max,
        vmin=time_min,
    )
    cb2 = fig2.colorbar(p2)

    def plot_it(n_cut, dpi=300):
        """
        Parameters
        ----------
        n_cut: int
            The number of the cut.
        dpi: int
            Dots per inch in the plot.
        """
        ax11.clear()
        ax11.plot_surface(ages_m, months_m, times,
            cmap=plt.cm.viridis,
            vmax=time_max,
            vmin=time_min,
            rstride=1, cstride=1, linewidth=0)
        ax11.set_xlabel('age')
        ax11.set_ylabel('month')
        ax11.set_zlabel('wake up time')
       ax11.set_xlim3d(age_min, age_max)
        ax11.set_ylim3d(month_min, month_max)
        ax11.set_zlim3d(time_min, time_max)

        p1 = ax12.pcolor(
            ages_m, months_m, times,
            cmap=plt.cm.viridis,
            vmax=time_max,
            vmin=time_min,
        )

        ax12.set_xlabel('age')
        ax12.set_ylabel('month')
        ax12.set_xlim(age_min, age_max)
        ax12.set_ylim(month_min, month_max)

        ax21.clear()
        ax21.plot_surface(ages_hi_m, months_hi_m, model,
            cmap=plt.cm.viridis,
            vmax=time_max,
            vmin=time_min,
            rstride=1, cstride=1, linewidth=0)

        ax21.set_xlabel('age')
        ax21.set_ylabel('month')
        ax21.set_zlabel('wake up time')
        ax21.set_xlim3d(age_min, age_max)
        ax21.set_ylim3d(month_min, month_max)
        ax21.set_zlim3d(time_min, time_max)

        p2 = ax22.pcolor(
            ages_hi_m, months_hi_m, model,
            cmap=plt.cm.viridis,
            vmax=time_max,
            vmin=time_min,
        )
        ax22.set_xlabel('age')
        ax22.set_ylabel('month')
        ax22.set_xlim(age_min, age_max)
        ax22.set_ylim(month_min, month_max)

        fig1.savefig('data_{}.png'.format(n_cut), dpi=dpi)
        fig2.savefig('model_{}.png'.format(n_cut), dpi=dpi)

    plot_it(0)

    # First cut
    cut1 = 32
    model[np.where(ages_hi_m < cut1)] = 7.1
    model[np.where(ages_hi_m >= cut1)] = 6.2

    ax12.plot(
        [cut1, cut1],
        [month_min, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )
    ax22.plot(
        [cut1, cut1],
        [month_min, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )
    plot_it(1)

    # Second cut
    cut2 = 9.5  # months
    model[np.where(ages_hi_m < cut1)] = 6.9
    model[np.where(np.logical_and(ages_hi_m < cut1, months_hi_m > cut2))] = 7.5
    ax12.plot(
        [age_min, cut1],
        [cut2, cut2],
        color='black',
        linewidth=2,
        zorder=3,
    )
    ax22.plot(
        [age_min, cut1],
        [cut2, cut2],
        color='black',
        linewidth=2,
        zorder=3,
    )

    plot_it(2)
   
    # Third cut
    cut3 = 3.5  # months
    model[np.where(np.logical_and(ages_hi_m < cut1, months_hi_m < cut3))] = 7.5
    ax12.plot(
        [age_min, cut1],
        [cut3, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )
    ax22.plot(
        [age_min, cut1],
        [cut3, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )

    plot_it(3)

    # Fourth cut
    cut4 = 48  # years
    model[np.where(ages_hi_m >= cut1)] = 6.4
    model[np.where(ages_hi_m >= cut4)] = 6.1

    ax12.plot(
        [cut4, cut4],
        [month_min, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )
    ax22.plot(
        [cut4, cut4],
        [month_min, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )

    plot_it(4)

    # Fifth cut
    cut5 = 18  # years
    model[np.where(np.logical_and(ages_hi_m < cut1, months_hi_m < cut3))] = 6.8
    model[np.where(np.logical_and(ages_hi_m < cut5, months_hi_m < cut3))] = 7.9

    ax12.plot(
        [cut5, cut5],
        [month_min, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )   
    ax22.plot( 
        [cut5, cut5],
        [month_min, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )   
        
    plot_it(5)
        
    # Sixth cut
    cut6 = 18  # years
    model[np.where(np.logical_and(ages_hi_m < cut1, months_hi_m > cut2))] = 6.8
    model[np.where(np.logical_and(ages_hi_m < cut6, months_hi_m > cut2))] = 7.9

    ax12.plot(
        [cut6, cut6],
        [cut2, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )   
    ax22.plot(
        [cut6, cut6],
        [cut2, month_max],
        color='black',
        linewidth=2,
        zorder=3,
    )   
        
    plot_it(6)

    # Seventh cut
    cut7 = 13  # years
    model[np.where(np.logical_and(
        ages_hi_m < cut1, np.logical_and(
            months_hi_m >= cut3, months_hi_m <= cut2)))] = 6.6
    model[np.where(np.logical_and(
        ages_hi_m < cut7, np.logical_and(
            months_hi_m >= cut3, months_hi_m <= cut2)))] = 7.2

    ax12.plot(
        [cut7, cut7],
        [cut2, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )
    ax22.plot(
        [cut7, cut7],
        [cut2, cut3],
        color='black',
        linewidth=2,
        zorder=3,
    )

    plot_it(7)


if __name__ == '__main__':
    main()                        
