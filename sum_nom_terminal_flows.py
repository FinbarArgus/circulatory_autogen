import pandas as pd
import re

# This codes isn't automated, the indexes for terminals were chosen from inspection of the df_vessels file

nom_flow_data = pd.read_csv('data/df_vessels.csv')

trunk_terminals = nom_flow_data['vessel_nz'][:23]
print('original terminals that merge into trunk_terminals are')
print(trunk_terminals)

leg_terminals = nom_flow_data['vessel_nz'][23:31]
leg_L_terminals = leg_terminals[leg_terminals.str.endswith('L')]
print('original terminals that merge into leg_L_terminals are')
print(leg_L_terminals)

leg_R_terminals = leg_terminals[leg_terminals.str.endswith('R')]
print('original terminals that merge into leg_R_terminals are')
print(leg_R_terminals)

arm_terminals = nom_flow_data['vessel_nz'][31:37]
arm_L_terminals = arm_terminals[arm_terminals.str.endswith('L')]
print('original terminals that merge into arm_L_terminals are')
print(arm_L_terminals)

arm_R_terminals = arm_terminals[arm_terminals.str.endswith('R')]
print('original terminals that merge into arm_R_terminals are')
print(arm_R_terminals)

head_terminals = nom_flow_data['vessel_nz'][37:]
head_L_terminals = head_terminals[head_terminals.str.endswith('L')]
print('original terminals that merge into head_L_terminals are')
print(head_L_terminals)

head_R_terminals = head_terminals[head_terminals.str.endswith('R')]
print('original terminals that merge into head_R_terminals are')
print(head_R_terminals)

external_carotid_terminals = nom_flow_data['vessel_nz'][37:39]
external_carotid_L_terminals = external_carotid_terminals[external_carotid_terminals.str.endswith('L')]
print('original terminals that merge into external_carotid_L_terminals are')
print(external_carotid_L_terminals)

external_carotid_R_terminals = external_carotid_terminals[external_carotid_terminals.str.endswith('R')]
print('original terminals that merge into external_carotid_R_terminals are')
print(external_carotid_R_terminals)

posterior_cerebral_terminals = nom_flow_data['vessel_nz'][39:45]
posterior_cerebral_L_terminals = posterior_cerebral_terminals[posterior_cerebral_terminals.str.endswith('L')]
print('original terminals that merge into posterior_cerebral_L_terminals are')
print(posterior_cerebral_L_terminals)

posterior_cerebral_R_terminals = posterior_cerebral_terminals[posterior_cerebral_terminals.str.endswith('R')]
print('original terminals that merge into posterior_cerebral_R_terminals are')
print(posterior_cerebral_R_terminals)

middle_cerebral_terminals = nom_flow_data['vessel_nz'][45:73]
middle_cerebral_L_terminals = middle_cerebral_terminals[middle_cerebral_terminals.str.endswith('L')]
print('original terminals that merge into middle_cerebral_L_terminals are')
print(middle_cerebral_L_terminals)

middle_cerebral_R_terminals = middle_cerebral_terminals[middle_cerebral_terminals.str.endswith('R')]
print('original terminals that merge into middle_cerebral_R_terminals are')
print(middle_cerebral_R_terminals)

anterior_cerebral_terminals = nom_flow_data['vessel_nz'][73:]
anterior_cerebral_L_terminals = anterior_cerebral_terminals[anterior_cerebral_terminals.str.endswith('L')]
print('original terminals that merge into anterior_cerebral_L_terminals are')
print(anterior_cerebral_L_terminals)

anterior_cerebral_R_terminals = anterior_cerebral_terminals[anterior_cerebral_terminals.str.endswith('R')]
print('original terminals that merge into anterior_cerebral_R_terminals are')
print(anterior_cerebral_R_terminals)

# sum flows from refined terminals to calulate nominal flow in new terminals
# note the flow is in cm^3/s
trunk_flow = 0
arm_L_flow = 0
arm_R_flow = 0
leg_L_flow = 0
leg_R_flow = 0
head_L_flow = 0
head_R_flow = 0
external_carotid_L_flow = 0
external_carotid_R_flow = 0
posterior_cerebral_L_flow = 0
posterior_cerebral_R_flow = 0
middle_cerebral_L_flow = 0
middle_cerebral_R_flow = 0
anterior_cerebral_L_flow = 0
anterior_cerebral_R_flow = 0
total_flow = 0
for II in range(nom_flow_data.shape[0]):
    vessel_name = nom_flow_data.iloc[II, 0]
    if any(trunk_terminals.str.contains(vessel_name)):
        trunk_flow += nom_flow_data.iloc[II, 3]
    if any(leg_L_terminals.str.contains(vessel_name)):
        leg_L_flow += nom_flow_data.iloc[II, 3]
    if any(leg_R_terminals.str.contains(vessel_name)):
        leg_R_flow += nom_flow_data.iloc[II, 3]
    if any(arm_L_terminals.str.contains(vessel_name)):
        arm_L_flow += nom_flow_data.iloc[II, 3]
    if any(arm_R_terminals.str.contains(vessel_name)):
        arm_R_flow += nom_flow_data.iloc[II, 3]
    if any(head_L_terminals.str.contains(vessel_name)):
        head_L_flow += nom_flow_data.iloc[II, 3]
    if any(head_R_terminals.str.contains(vessel_name)):
        head_R_flow += nom_flow_data.iloc[II, 3]
    if any(external_carotid_L_terminals.str.contains(vessel_name)):
        external_carotid_L_flow += nom_flow_data.iloc[II, 3]
    if any(external_carotid_R_terminals.str.contains(vessel_name)):
        external_carotid_R_flow += nom_flow_data.iloc[II, 3]
    if any(posterior_cerebral_L_terminals.str.contains(vessel_name)):
        posterior_cerebral_L_flow += nom_flow_data.iloc[II, 3]
    if any(posterior_cerebral_R_terminals.str.contains(vessel_name)):
        posterior_cerebral_R_flow += nom_flow_data.iloc[II, 3]
    if any(middle_cerebral_L_terminals.str.contains(vessel_name)):
        middle_cerebral_L_flow += nom_flow_data.iloc[II, 3]
    if any(middle_cerebral_R_terminals.str.contains(vessel_name)):
        middle_cerebral_R_flow += nom_flow_data.iloc[II, 3]
    if any(anterior_cerebral_L_terminals.str.contains(vessel_name)):
        anterior_cerebral_L_flow += nom_flow_data.iloc[II, 3]
    if any(anterior_cerebral_R_terminals.str.contains(vessel_name)):
        anterior_cerebral_R_flow += nom_flow_data.iloc[II, 3]
    total_flow += nom_flow_data.iloc[II, 3]

print(f'trunk_flow                 =   {trunk_flow}')
print(f'leg_L_flow                 =   {leg_L_flow}')
print(f'leg_R_flow                 =   {leg_R_flow}')
print(f'arm_L_flow                 =   {arm_L_flow}')
print(f'arm_R_flow                 =   {arm_R_flow}')
print(f'head_L_flow                =   {head_L_flow}')
print(f'head_R_flow                =   {head_R_flow}')
print(f'external_carotid_L_flow    =   {external_carotid_L_flow}')
print(f'external_carotid_R_flow    =   {external_carotid_R_flow}')
print(f'posterior_cerebral_L_flow  =   {posterior_cerebral_L_flow}')
print(f'posterior_cerebral_R_flow  =   {posterior_cerebral_R_flow}')
print(f'middle_cerebral_L_flow     =   {middle_cerebral_L_flow}')
print(f'middle_cerebral_R_flow     =   {middle_cerebral_R_flow}')
print(f'anterior_cerebral_L_flow   =   {anterior_cerebral_L_flow}')
print(f'anterior_cerebral_R_flow   =   {anterior_cerebral_R_flow}')

head_check = head_L_flow + head_R_flow - external_carotid_L_flow - external_carotid_R_flow - \
    posterior_cerebral_L_flow - posterior_cerebral_R_flow - middle_cerebral_L_flow - \
    middle_cerebral_R_flow - anterior_cerebral_L_flow - anterior_cerebral_R_flow


total_check = total_flow - trunk_flow - arm_L_flow - arm_R_flow - leg_L_flow - leg_R_flow \
              - external_carotid_L_flow - external_carotid_R_flow - \
             posterior_cerebral_L_flow - posterior_cerebral_R_flow - middle_cerebral_L_flow - \
             middle_cerebral_R_flow - anterior_cerebral_L_flow - anterior_cerebral_R_flow

print(f'head flow calc check. This should be zero : {head_check}')
print(f'total flow calc check. This should be zero : {total_check}')



