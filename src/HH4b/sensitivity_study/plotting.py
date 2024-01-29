import uproot
import ROOT as rt

def plot_2D(h, output, sample, var, cut, lumi, plot_dir, plot_name, xaxis, yaxis, zaxis, logz=False, normed=False, print_integral=False):
    """
    Function to plot 2D histograms
    """
    c = rt.TCanvas('c','c', 800, 800)
    rt.gStyle.SetOptStat(0)
    rt.gStyle.SetOptTitle(0)
    h.SetTitle("")
    if normed:
        h.Scale(1./h.Integral())
    if logz:
        c.SetLogz()
    h.GetXaxis().SetTitle(xaxis)
    h.GetYaxis().SetTitle(yaxis)
    h.GetZaxis().SetTitle(zaxis)
    h.Draw("COLZ")
    c.SaveAs(f"{plot_dir}/{plot_name}.png")
    c.SaveAs(f"{plot_dir}/{plot_name}.pdf")
    return

def plot_data_mc(data_file_path, mc_file_path, plot_dir):
    # Load the data and MC root files
    data_file = uproot.open(data_file_path)
    mc_file = uproot.open(mc_file_path)

    # Extract the histograms from the root files
    data_hist = data_file["YOUR_HISTOGRAM_NAME"]  # Replace with your histogram name
    mc_hist = mc_file["YOUR_HISTOGRAM_NAME"]  # Replace with your histogram name

    # Plot the Data histogram
    plot_2D(data_hist, None, "Data", "var", "cut", 60.0, plot_dir, "data_plot", "X-axis label", "Y-axis label", "Z-axis label")

    # Plot the MC histogram
    plot_2D(mc_hist, None, "MC", "var", "cut", 60.0, plot_dir, "mc_plot", "X-axis label", "Y-axis label", "Z-axis label")

if __name__ == "__main__":
    data_file_path = "/path/to/your/data/root/file.root"  # Replace with your data file path
    mc_file_path = "/path/to/your/mc/root/file.root"  # Replace with your MC file path
    plot_dir = "/path/to/save/plots"  # Replace with the directory where you want to save the plots

    plot_data_mc(data_file_path, mc_file_path, plot_dir)
