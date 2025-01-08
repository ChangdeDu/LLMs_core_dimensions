import seaborn as sns
import matplotlib as mpl
from scipy.io import loadmat
import matplotlib.pyplot as plt

subject_list =[1,2,5,7]
for subj in subject_list:

    mat_file_path = 'data/ROIs/Subject_' + str(subj) + '/fmri_shared_1k_all_rois_3d_R2_gemini.mat'
    mat_contents = loadmat(mat_file_path)
    test_data = mat_contents['data']
    flattened_pcc_mllm = test_data.ravel()

    mat_file_path = 'data/ROIs/Subject_' + str(subj) + '/fmri_shared_1k_all_rois_3d_R2_chatgpt.mat'
    mat_contents = loadmat(mat_file_path)
    test_data = mat_contents['data']
    flattened_pcc_llm = test_data.ravel()

    mat_file_path = 'data/ROIs/Subject_' + str(subj) + '/fmri_shared_1k_all_rois_3d_R2_humans.mat'
    mat_contents = loadmat(mat_file_path)
    test_data = mat_contents['data']
    flattened_pcc_human = test_data.ravel()

    plt.figure()
    x = [-0.05, 1]
    y = [-0.05, 1]
    w = [-0.05, 0.85]
    sns.lineplot(x=x, y=y, linewidth=3.5, color="red", label="human level")
    sns.lineplot( x=x, y=w, linewidth=3.5, color="orange",  linestyle="--", label="85% human level")

    plt.hist2d(
        flattened_pcc_human,
        flattened_pcc_llm,
        bins=100,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
    )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel("Human Performance $(R^2)$", fontsize=26)
    plt.ylabel("LLM Performance $(R^2)$", fontsize=26)
    plt.title("Voxel-wise Encoding", fontsize=26)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.savefig('figures/2dhist_llm_vs_human_'+str(subj)+'_lr_R2.pdf', bbox_inches='tight', pad_inches=0)

    plt.figure()
    x = [-0.05, 1]
    y = [-0.05, 1]
    w = [-0.05, 0.85]
    sns.lineplot(x=x, y=y, linewidth=3.5, color="red", label="human level")
    sns.lineplot( x=x, y=w, linewidth=3.5, color="orange",  linestyle="--", label="85% human level")

    plt.hist2d(
        flattened_pcc_human,
        flattened_pcc_mllm,
        bins=100,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
    )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel("Human Performance $(R^2)$", fontsize=26)
    plt.ylabel("MLLM Performance $(R^2)$", fontsize=26)
    plt.title("Voxel-wise Encoding", fontsize=26)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=20)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.savefig('figures/2dhist_mllm_vs_human_'+str(subj)+'_lr_R2.pdf', bbox_inches='tight', pad_inches=0)