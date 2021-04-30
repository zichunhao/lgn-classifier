import torch
import numpy as np

def load_pt_file(filename, path='./150p/'):
    path = path + filename
    print(f"loading {path}...")
    data = torch.load(path)
    return data

# [eta, phi, pt, tag] -> [E, px, py, pz, p, tag]
def cartesian(p_list):
    eta, phi, pt, tag = p_list
    if tag > 0: # real data
        tag = 1
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(2) * pt * np.cosh(eta)
    else: # padded data
        tag = 0
        px = py = pz = E = 0
    return [E, px, py, pz, tag]

# convert the whole data set
def convert_to_cartesian(jet_data, name_str, save=False):
    print(f"reformatting {name_str}...")

    shape = list(jet_data.shape)
    shape[-1] += 1 # [eta, phi, pt, tag] -> [E, px, py, pz, tag]
    shape = tuple(shape) # (_,_,4)

    jet_data_cartesian = np.zeros(shape)

    for jet in range(len(jet_data)):
        for particle in range(len(jet_data[jet])):
            jet_data_cartesian[jet][particle] = cartesian(jet_data[jet][particle])

    if save:
        print(f"saving {name_str}...")
        filename = name_str + '_cartesian.pt'
        path = './150p/cartesian/' + filename
        torch.save(jet_data_cartesian, path)
        print(f"{name_str} saved as {path}")

    return jet_data_cartesian

# data loading
all_g_jets_150p_polarrel_mask = load_pt_file('all_g_jets_150p_polarrel_mask.pt', path='./150p/mask/').numpy()
all_q_jets_150p_polarrel_mask = load_pt_file('all_q_jets_150p_polarrel_mask.pt', path='./150p/mask/').numpy()
all_t_jets_150p_polarrel_mask = load_pt_file('all_t_jets_150p_polarrel_mask.pt', path='./150p/mask/').numpy()
all_w_jets_150p_polarrel_mask = load_pt_file('all_w_jets_150p_polarrel_mask.pt', path='./150p/mask/').numpy()
all_z_jets_150p_polarrel_mask = load_pt_file('all_z_jets_150p_polarrel_mask.pt', path='./150p/mask/').numpy()

# converting to cartesian coordinates: [eta, phi, pt, tag] -> [E, px, py, pz, tag]
all_g_jets_150p_polarrel_mask_cartesian = convert_to_cartesian(all_g_jets_150p_polarrel_mask, "all_g_jets_150p_polarrel_mask")
all_q_jets_150p_polarrel_mask_cartesian = convert_to_cartesian(all_q_jets_150p_polarrel_mask, "all_q_jets_150p_polarrel_mask")
all_t_jets_150p_polarrel_mask_cartesian = convert_to_cartesian(all_t_jets_150p_polarrel_mask, "all_t_jets_150p_polarrel_mask")
all_w_jets_150p_polarrel_mask_cartesian = convert_to_cartesian(all_w_jets_150p_polarrel_mask, "all_w_jets_150p_polarrel_mask")
all_z_jets_150p_polarrel_mask_cartesian = convert_to_cartesian(all_z_jets_150p_polarrel_mask, "all_z_jets_150p_polarrel_mask")

# particle 4-momenta: [E, px, py, pz]
g_p4_cartesian = torch.from_numpy(all_g_jets_150p_polarrel_mask_cartesian[:,:,:4])
q_p4_cartesian = torch.from_numpy(all_q_jets_150p_polarrel_mask_cartesian[:,:,:4])
t_p4_cartesian = torch.from_numpy(all_t_jets_150p_polarrel_mask_cartesian[:,:,:4])
w_p4_cartesian = torch.from_numpy(all_w_jets_150p_polarrel_mask_cartesian[:,:,:4])
z_p4_cartesian = torch.from_numpy(all_z_jets_150p_polarrel_mask_cartesian[:,:,:4])

p4_cartesian = torch.cat((g_p4_cartesian, q_p4_cartesian, t_p4_cartesian, w_p4_cartesian, z_p4_cartesian),0)

# # particle masses: p
# g_scalar = torch.from_numpy(all_g_jets_150p_polarrel_mask_cartesian[:,:,:-2]).unsqueeze(-1)
# q_scalar = torch.from_numpy(all_q_jets_150p_polarrel_mask_cartesian[:,:,:-2]).unsqueeze(-1)
# t_scalar = torch.from_numpy(all_t_jets_150p_polarrel_mask_cartesian[:,:,:-2]).unsqueeze(-1)
# w_scalar = torch.from_numpy(all_w_jets_150p_polarrel_mask_cartesian[:,:,:-2]).unsqueeze(-1)
# z_scalar = torch.from_numpy(all_z_jets_150p_polarrel_mask_cartesian[:,:,:-2]).unsqueeze(-1)
#
# scalars = torch.cat((g_scalar, q_scalar, t_scalar, w_scalar, z_scalar),0)

# jet types
# g_tag: [1, 0, 0, 0, 0]
# q_tag: [0, 1, 0, 0, 0]
# t_tag: [0, 0, 1, 0, 0]
# w_tag: [0, 0, 0, 1, 0]
# z_tag: [0, 0, 0, 0, 1]
g_tag = torch.tensor([1, 0, 0, 0, 0])
q_tag = torch.tensor([0, 1, 0, 0, 0])
t_tag = torch.tensor([0, 0, 1, 0, 0])
w_tag = torch.tensor([0, 0, 0, 1, 0])
z_tag = torch.tensor([0, 0, 0, 0, 1])

g_tags = torch.cat([g_tag]*g_p4_cartesian.shape[0]).view(-1,5)
q_tags = torch.cat([q_tag]*q_p4_cartesian.shape[0]).view(-1,5)
t_tags = torch.cat([t_tag]*t_p4_cartesian.shape[0]).view(-1,5)
w_tags = torch.cat([w_tag]*w_p4_cartesian.shape[0]).view(-1,5)
z_tags = torch.cat([z_tag]*z_p4_cartesian.shape[0]).view(-1,5)

tags = torch.cat((g_tags, q_tags, t_tags, w_tags, z_tags),0)

# labels: 1 if real data and 0 if padded data
g_label = torch.from_numpy(all_g_jets_150p_polarrel_mask_cartesian[:,:,-1])
q_label = torch.from_numpy(all_q_jets_150p_polarrel_mask_cartesian[:,:,-1])
t_label = torch.from_numpy(all_t_jets_150p_polarrel_mask_cartesian[:,:,-1])
w_label = torch.from_numpy(all_w_jets_150p_polarrel_mask_cartesian[:,:,-1])
z_label = torch.from_numpy(all_z_jets_150p_polarrel_mask_cartesian[:,:,-1])

labels = torch.cat((g_label, q_label, t_label, w_label, z_label),0)

# number of particles per jet
Nobj = labels.sum(dim=-1)

# # masks
# node_mask = data['p4'][...,0] != 0
# node_mask = node_mask.to(torch.uint8)
# edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

# creating a dictionary
data = {}

data['jet_types'] = tags  # g/q/t/w/z
data['Nobj'] = Nobj  # number of particles (non-padded) per jet
data['labels'] = labels  # 1 if real data and 0 if padded data
data['p4'] = p4_cartesian  # particle-level 4-momenta
# data['node_mask'] = node_mask
# data['edge_mask'] = edge_mask
# data['scalars'] = scalars  # particle masses in each jet


# save model
torch.save(data, './hls4ml.pt')
