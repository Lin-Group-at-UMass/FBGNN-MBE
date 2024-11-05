import copy
from file_utils import ensure_path_exists
from tool_utils import load_daset_config
        
def make_comp_body(t, start_i, two_body_num, three_body_num):
    comp2idx_2b = dict()
    comp2idx_3b = dict()
    idx2comp_2b = dict()
    idx2comp_3b = dict()

    for i in range(start_i, t + 1):
        for j in range(i + 1, t + 1):
            if i not in comp2idx_2b:
                comp2idx_2b[i] = dict()
            comp2idx_2b[i][j] = two_body_num
            idx2comp_2b[two_body_num] = [i, j]
            two_body_num += 1
            for k in range(j + 1, t + 1):
                if i not in comp2idx_3b:
                    comp2idx_3b[i] = dict()
                if j not in comp2idx_3b[i]:
                    comp2idx_3b[i][j] = dict()
                comp2idx_3b[i][j][k] = three_body_num
                idx2comp_3b[three_body_num] = [i, j, k]
                three_body_num += 1
    return idx2comp_2b, idx2comp_3b, comp2idx_2b

def make_mixed_comp_body():
    t = 20
    two_body_num = 1
    three_body_num = 1
    start_i = 1
    return make_comp_body(t, start_i, two_body_num, three_body_num)

def make_phoh_comp_body():
    t = 9 # 10 in total, 9 as inside make_comp_body, range(start_i, t+1)
    two_body_num = 0
    three_body_num = 0
    start_i = 0
    return make_comp_body(t, start_i, two_body_num, three_body_num)

def make_h2o_comp_body():
    t = 66 # 67 in total, 32 as inside make_comp_body, range(start_i, t+1)
    two_body_num = 0
    three_body_num = 0
    start_i = 0
    return make_comp_body(t, start_i, two_body_num, three_body_num)

def get_k_body_energy(path):
    one_body_energy = dict()
    with open(path) as one_body_result:
        l_onebody = one_body_result.readlines()
        for line in l_onebody:
            line = line.strip('\n')
            energy = line.split(',')[1]
            
            idx_keys = line.split(',')[0].split('_')
            
            idx = 0
            water_num = 0
            
            if len(idx_keys) == 3:
                idx = int(idx_keys[2])
                water_num = int(idx_keys[1])
            elif len(idx_keys) == 2:
                idx = int(idx_keys[1])
                water_num = int(idx_keys[0])
            else:
                raise ValueError('Invalid key')
            
            if water_num not in one_body_energy:
                one_body_energy[water_num] = dict()
            one_body_energy[water_num][idx] = eval(energy)
    return one_body_energy


def make_energy_file(two_body_energy, one_body_energy, idx2comp_2b, filename):
    ensure_path_exists(filename)
    two_body_energy_new = copy.deepcopy(two_body_energy)
    
    print(len(two_body_energy), len(two_body_energy))
    
    with open(filename, 'w') as f:
        for molecule_num in two_body_energy:
            for idx in two_body_energy[molecule_num]:
                energy_eij = two_body_energy[molecule_num][idx]
                i, j = idx2comp_2b[idx][0], idx2comp_2b[idx][1]
                energy = energy_eij - one_body_energy[molecule_num][i] - one_body_energy[molecule_num][j]
                two_body_energy_new[molecule_num][idx] = energy
                content = f'2_{molecule_num}_{idx},{energy}\n'
                f.write(content)
    f.close()
    return two_body_energy_new
    

def make_3body_energy_file(three_body_energy, two_body_energy_new, one_body_energy, idx2comp_3b, comp2idx_2b, filename):
    ensure_path_exists(filename)
    
    three_body_energy_new = copy.deepcopy(three_body_energy)
    
    count = 0
    with open(filename, 'w') as f:
        for molecule_num in three_body_energy:
            for idx in three_body_energy[molecule_num]:
                energy_eijk = three_body_energy[molecule_num][idx]
                i, j, k = idx2comp_3b[idx][0], idx2comp_3b[idx][1], idx2comp_3b[idx][2]
                i_2b, j_2b, k_2b = comp2idx_2b[i][j], comp2idx_2b[i][k], comp2idx_2b[j][k]

                try:
                    energy = energy_eijk - \
                         one_body_energy[molecule_num][i] - one_body_energy[molecule_num][j] - \
                         one_body_energy[molecule_num][k] - two_body_energy_new[molecule_num][i_2b] - \
                         two_body_energy_new[molecule_num][j_2b] - two_body_energy_new[molecule_num][k_2b]
                    three_body_energy_new[molecule_num][idx] = energy
                    three_body_energy_new[molecule_num][idx] = energy
                    content = f'3_{molecule_num}_{idx},{energy}\n'
                    f.write(content)
                    count += 1
                except:
                    pass
    f.close()
    print(count)
   
   
def make_energies(cfig):    
    s_path = cfig['source_path']
    
    total = cfig['total_moleculars']
    start_i = cfig['start_i']
    two_body_num = cfig['two_body_num']
    three_body_num = cfig['three_body_num']
    
    if "dest_path" in cfig.keys():
        d_path = cfig['dest_path']
    else:
        d_path = s_path
    
        
    idx2comp_2b, idx2comp_3b, comp2idx_2b = make_comp_body(total, start_i, two_body_num, three_body_num)
    one_body_energy = get_k_body_energy(f'{s_path}/1body_result')
    two_body_energy = get_k_body_energy(f'{s_path}/2body_result')
    three_body_energy = get_k_body_energy(f'{s_path}/3body_result')
    
    two_body_energy_new = make_energy_file(two_body_energy, one_body_energy, idx2comp_2b, f'{d_path}/2body_energy.txt')
    
    make_3body_energy_file(three_body_energy, two_body_energy_new, one_body_energy, idx2comp_3b, comp2idx_2b, f'{d_path}/3body_energy.txt')
         
    
if __name__ == '__main__':
    d_configs = load_daset_config()
    for key in d_configs.keys():
        make_energies(d_configs[key])
