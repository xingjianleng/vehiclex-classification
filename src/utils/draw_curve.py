import matplotlib.pyplot as plt
import seaborn as sns


def draw_curve(path, x_epoch, train_loss, test_loss, train_result, test_result):
    # draw curve for loss and result during training and testing iterations
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    sns.lineplot(x=x_epoch, y=train_loss, ax=ax[0], color='blue', marker='o', label='train' + ': {:.3f}'.format(train_loss[-1]))
    sns.lineplot(x=x_epoch, y=test_loss, ax=ax[0], color='red', marker='o', label='test' + ': {:.3f}'.format(test_loss[-1]))

    sns.lineplot(x=x_epoch, y=train_result, ax=ax[1], color='blue', marker='o', label='train' + ': {:.1f}'.format(train_result[-1]))
    sns.lineplot(x=x_epoch, y=test_result, ax=ax[1], color='red', marker='o', label='test' + ': {:.1f}'.format(test_result[-1]))

    ax[0].set_title('loss')
    ax[1].set_title('result in %')

    ax[0].legend()
    ax[1].legend()

    fig.savefig(path)
    plt.close(fig)
