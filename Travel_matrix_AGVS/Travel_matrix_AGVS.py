"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import itertools
import random
from collections import defaultdict


import numpy as np
import pandas as pd




# Machine information
machinesPerWC = [4, 2, 5, 3, 2]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]  # Index of machines

# Central buffer information
noOfCbPerWC = [1, 1, 1, 1, 1]
central_buffers_cap_WC = [[1], [1], [1], [1], [1]]  # buffer cap per central buffer

# Number of Work Centers
noOfWC = range(len(machinesPerWC))



def choose_distance_matrix(AGV_selection, agvsPerWC_input):
    travel_time_matrix_static = {
        'depot': {(0, 0): 0.8368268315517269, (1, 0): 1.3789959269986793, (2, 0): 0.7374943418267287,
                  (3, 0): 1.2972283538880491, (0, 1): 0.8505186540501775, (1, 1): 0.5416771040851046,
                  (0, 2): 1.4821692252025425, (1, 2): 0.7220310992016531, (2, 2): 1.44015313750998,
                  (3, 2): 0.9980913574974206, (4, 2): 1.4655973804643136, (0, 3): 0.6900408716161083,
                  (1, 3): 1.156939287064361, (2, 3): 1.4497552920043493, (0, 4): 1.3565973476745707,
                  (1, 4): 0.974565533688489, (4, 0): 1.394814347842095, (2, 1): 0.9999902645977072,
                  (5, 2): 1.4399313296454057, (3, 3): 1.0171963499759777, (2, 4): 1.4658774786362851, 'depot': 0},
        (0, 0): {(0, 0): 0, (1, 0): 1.26620237490732, (2, 0): 1.2685755503713474, (3, 0): 1.373931144150298,
                 (0, 1): 0.8385576112846133, (1, 1): 1.2220659868833823, (0, 2): 1.050221010541266,
                 (1, 2): 1.2213984178668447, (2, 2): 1.4655967707897908, (3, 2): 0.633680336813525,
                 (4, 2): 0.661954420471606, (0, 3): 0.5405576624988631, (1, 3): 0.9872534043303917,
                 (2, 3): 0.9637099435981364, (0, 4): 0.9084786574735028, (1, 4): 1.3170079398666712,
                 (4, 0): 1.3171978338148498, (2, 1): 0.9048601024329098, (5, 2): 1.2006267618969637,
                 (3, 3): 1.2697460664386844, (2, 4): 1.41861561946562, 'depot': 1.1487287241765198},
        (1, 0): {(0, 0): 0.7669892737945969, (1, 0): 0, (2, 0): 1.3303484608036302, (3, 0): 0.7330989396314361,
                 (0, 1): 1.0491718272524513, (1, 1): 1.3248312441627579, (0, 2): 1.1420050602457241,
                 (1, 2): 1.0663696069763096, (2, 2): 1.0069879277611902, (3, 2): 1.0707765811139016,
                 (4, 2): 0.929180865016188, (0, 3): 0.788995120663737, (1, 3): 1.24458415572796,
                 (2, 3): 1.127208994765813, (0, 4): 0.6516456195126231, (1, 4): 0.5928333903384416,
                 (4, 0): 0.5156001379404435, (2, 1): 0.5606362504827016, (5, 2): 0.9116714884584679,
                 (3, 3): 0.8713917524447744, (2, 4): 0.5797066116059516, 'depot': 0.8696714400565698},
        (2, 0): {(0, 0): 1.4283219310526878, (1, 0): 1.4120705154082192, (2, 0): 0, (3, 0): 0.6664134828591556,
                 (0, 1): 0.8513164737770218, (1, 1): 0.5315966169022929, (0, 2): 0.9092392757050257,
                 (1, 2): 0.5045408582777703, (2, 2): 1.018285822170629, (3, 2): 1.418419527183109,
                 (4, 2): 1.3991852061343057, (0, 3): 1.380608948576235, (1, 3): 1.4478134829110432,
                 (2, 3): 0.8930683140588253, (0, 4): 0.5195092075373976, (1, 4): 1.3874423072960376,
                 (4, 0): 0.7344923163091427, (2, 1): 0.941257709811324, (5, 2): 1.3354257686899214,
                 (3, 3): 0.6454551019572973, (2, 4): 1.060137721957045, 'depot': 1.358089969570941},
        (3, 0): {(0, 0): 0.6591452175542133, (1, 0): 0.609776463316532, (2, 0): 1.282024280335028, (3, 0): 0,
                 (0, 1): 0.8318287196052903, (1, 1): 0.5322936834178769, (0, 2): 0.9095974593770558,
                 (1, 2): 0.8103956170760841, (2, 2): 1.125453202642194, (3, 2): 0.6249779196820321,
                 (4, 2): 1.1743093641147841, (0, 3): 1.3587086554538987, (1, 3): 1.458200363716748,
                 (2, 3): 0.7787133691702512, (0, 4): 1.205098468936339, (1, 4): 1.226943051469723,
                 (4, 0): 1.309091394172805, (2, 1): 0.6822370587609035, (5, 2): 0.5630647737114909,
                 (3, 3): 1.331047917780142, (2, 4): 1.056343644024528, 'depot': 1.4089047358945168},
        (0, 1): {(0, 0): 0.6481618310687073, (1, 0): 0.6113151271351946, (2, 0): 0.6604945325389536,
                 (3, 0): 1.2153990517792432, (0, 1): 0, (1, 1): 0.8712937924466125, (0, 2): 0.6683126628454027,
                 (1, 2): 0.5250295689249838, (2, 2): 0.7923312086031519, (3, 2): 1.2981015284328743,
                 (4, 2): 1.4717069323945031, (0, 3): 0.7063276059205963, (1, 3): 0.9195381319168885,
                 (2, 3): 1.0180484701821073, (0, 4): 1.0328275289382778, (1, 4): 0.5020376455759603,
                 (4, 0): 1.1958650978685652, (2, 1): 0.8197745598966976, (5, 2): 0.5624762314920712,
                 (3, 3): 0.5868275087109212, (2, 4): 0.6892370306477282, 'depot': 0.6414097142120996},
        (1, 1): {(0, 0): 1.2080141984184922, (1, 0): 1.0990667208160936, (2, 0): 0.9299852037223443,
                 (3, 0): 0.711508576492168, (0, 1): 1.2822472278395933, (1, 1): 0, (0, 2): 0.615840838462136,
                 (1, 2): 0.9665187336339034, (2, 2): 0.7705113339358283, (3, 2): 0.5148877953506028,
                 (4, 2): 0.5045589930286621, (0, 3): 0.8120928197219694, (1, 3): 0.5164378477829402,
                 (2, 3): 0.5259242183165135, (0, 4): 1.4868846801362854, (1, 4): 1.029006610058782,
                 (4, 0): 1.2517100099273373, (2, 1): 0.7928123550737989, (5, 2): 0.6367076059064721,
                 (3, 3): 0.6465901838984887, (2, 4): 0.8771455682061517, 'depot': 1.108409768401753},
        (0, 2): {(0, 0): 0.7827862285324769, (1, 0): 0.9040067594426785, (2, 0): 0.6418280213299007,
                 (3, 0): 1.2379995204976124, (0, 1): 1.2635143947718175, (1, 1): 0.5896145638286947, (0, 2): 0,
                 (1, 2): 0.7838075048037065, (2, 2): 1.2061227090546733, (3, 2): 0.7705738176907695,
                 (4, 2): 0.9835592297955192, (0, 3): 0.6709437043418548, (1, 3): 0.9137305352934587,
                 (2, 3): 0.7076458154063727, (0, 4): 1.14818692001697, (1, 4): 0.7891322652874209,
                 (4, 0): 1.3649894163563023, (2, 1): 0.6141351932637927, (5, 2): 1.1414489189581303,
                 (3, 3): 0.8243769289282149, (2, 4): 1.4252761488381183, 'depot': 1.4301422436211928},
        (1, 2): {(0, 0): 0.6124541817951766, (1, 0): 1.3809675920662596, (2, 0): 1.3998512455084433,
                 (3, 0): 1.1354427830000526, (0, 1): 0.8225903102189775, (1, 1): 1.2126988684910067,
                 (0, 2): 0.9712400281581305, (1, 2): 0, (2, 2): 1.413040335451976, (3, 2): 1.493820688629641,
                 (4, 2): 0.6199696696875726, (0, 3): 0.6555071553177974, (1, 3): 1.0982828150235058,
                 (2, 3): 0.9019015811532337, (0, 4): 1.1500134656413807, (1, 4): 1.2348999341282259,
                 (4, 0): 0.9748495474781398, (2, 1): 0.67357464187546, (5, 2): 1.042390173243731,
                 (3, 3): 0.8557616097717844, (2, 4): 1.2838531766236363, 'depot': 1.4866893866063764},
        (2, 2): {(0, 0): 1.2388237933393307, (1, 0): 1.217611565640372, (2, 0): 1.3215764434761825,
                 (3, 0): 0.7427649954393222, (0, 1): 0.627223286527875, (1, 1): 0.6949887546800944,
                 (0, 2): 1.398531893687182, (1, 2): 0.8573558015121832, (2, 2): 0, (3, 2): 0.5830502375899107,
                 (4, 2): 0.7611213873789884, (0, 3): 0.501653278658597, (1, 3): 0.5444424119208822,
                 (2, 3): 0.6037945440547243, (0, 4): 1.4355592252199472, (1, 4): 0.8082154498513499,
                 (4, 0): 1.1429720572595787, (2, 1): 1.419228526646048, (5, 2): 0.6389441931756812,
                 (3, 3): 0.6003214601873224, (2, 4): 1.1051403983977237, 'depot': 1.19548940707088},
        (3, 2): {(0, 0): 0.9758196535361491, (1, 0): 1.293745355872567, (2, 0): 1.4010679646353945,
                 (3, 0): 0.5683198364864761, (0, 1): 1.2344994594502108, (1, 1): 0.8412312728027703,
                 (0, 2): 0.5334345956918481, (1, 2): 1.0911200364557747, (2, 2): 0.7004339166836784, (3, 2): 0,
                 (4, 2): 1.295445867285318, (0, 3): 0.9881717015522258, (1, 3): 1.1013488033839565,
                 (2, 3): 1.0772078144959645, (0, 4): 1.0787518926380346, (1, 4): 1.4595004284513546,
                 (4, 0): 0.6067791865762877, (2, 1): 1.1192325987765044, (5, 2): 1.3042202662533509,
                 (3, 3): 0.9884580643706739, (2, 4): 1.4327037759286, 'depot': 1.3368689366278312},
        (4, 2): {(0, 0): 0.9524949297748568, (1, 0): 0.7610161014194881, (2, 0): 0.5232797495784187,
                 (3, 0): 0.9971749511144058, (0, 1): 1.3003170052393351, (1, 1): 0.6680509852723505,
                 (0, 2): 1.1483064744840856, (1, 2): 0.6426327132185297, (2, 2): 1.3039605541412072,
                 (3, 2): 0.8853003793026597, (4, 2): 0, (0, 3): 0.5870470776266782, (1, 3): 0.9269707633284358,
                 (2, 3): 0.575307230197392, (0, 4): 0.7586126541890853, (1, 4): 1.4251682428361225,
                 (4, 0): 1.0392535246464794, (2, 1): 1.1915563680809689, (5, 2): 0.862421525173756,
                 (3, 3): 1.188314301671662, (2, 4): 0.9108420412262181, 'depot': 1.3601487438746833},
        (0, 3): {(0, 0): 1.4024068635154856, (1, 0): 0.8758351494344505, (2, 0): 0.6066715715587567,
                 (3, 0): 1.3439034874281166, (0, 1): 1.0093932075445848, (1, 1): 1.4715322646655113,
                 (0, 2): 1.1647604271589262, (1, 2): 1.4852902322145214, (2, 2): 0.6364097957851038,
                 (3, 2): 0.6579054586189026, (4, 2): 0.5178679999902122, (0, 3): 0, (1, 3): 1.173914598887294,
                 (2, 3): 0.7833109146531365, (0, 4): 0.7002003008125097, (1, 4): 0.7893238445808916,
                 (4, 0): 0.6989186911051378, (2, 1): 1.0352098044054021, (5, 2): 0.6593899896166383,
                 (3, 3): 1.1571784287988045, (2, 4): 1.080322645357759, 'depot': 0.7385336762559601},
        (1, 3): {(0, 0): 0.9287428871569698, (1, 0): 0.6281956678497107, (2, 0): 0.6337927038735722,
                 (3, 0): 1.2626381623159895, (0, 1): 0.5969494238207368, (1, 1): 0.7950955043332928,
                 (0, 2): 0.5077006027612488, (1, 2): 1.166555359864097, (2, 2): 0.79096378680531,
                 (3, 2): 1.0037892023109327, (4, 2): 1.2764261469549432, (0, 3): 0.5059543245212147, (1, 3): 0,
                 (2, 3): 0.7439924147334476, (0, 4): 0.7725225404817735, (1, 4): 1.4279556009753098,
                 (4, 0): 1.251509565645788, (2, 1): 0.7514362841431027, (5, 2): 0.6708942106880065,
                 (3, 3): 1.3120976901898442, (2, 4): 0.7955037577904766, 'depot': 0.5294546128438232},
        (2, 3): {(0, 0): 1.0693375232350257, (1, 0): 1.3473612281161722, (2, 0): 0.8913712138204395,
                 (3, 0): 1.1592887942908243, (0, 1): 1.310631600562735, (1, 1): 0.8814296194416363,
                 (0, 2): 0.8860348849889105, (1, 2): 1.0518939500561473, (2, 2): 0.9375063649809722,
                 (3, 2): 0.6340809789213395, (4, 2): 0.5232992626692129, (0, 3): 1.4356766100791556,
                 (1, 3): 1.3376503228559062, (2, 3): 0, (0, 4): 1.3582675070935095, (1, 4): 1.4799936633252342,
                 (4, 0): 0.9573188817523459, (2, 1): 1.2949292193761757, (5, 2): 0.6270248150285677,
                 (3, 3): 0.8422167692444285, (2, 4): 0.5666962882995573, 'depot': 1.26138277946686},
        (0, 4): {(0, 0): 1.018952752587917, (1, 0): 1.3601832529389712, (2, 0): 1.4394190516246246,
                 (3, 0): 0.8187724612936392, (0, 1): 1.1923806423942214, (1, 1): 0.6269320293528657,
                 (0, 2): 0.746698020981588, (1, 2): 0.6274161884305687, (2, 2): 0.5527998574080122,
                 (3, 2): 1.0653107182040267, (4, 2): 0.9778843793085464, (0, 3): 1.1198780565175506,
                 (1, 3): 0.7588362529057211, (2, 3): 1.3568156540776783, (0, 4): 0, (1, 4): 0.9074709786783142,
                 (4, 0): 1.0850747103141611, (2, 1): 0.5367831740595418, (5, 2): 1.311341806609529,
                 (3, 3): 1.2737512567652172, (2, 4): 1.0950552811920484, 'depot': 0.5508569323449957},
        (1, 4): {(0, 0): 0.5162372662823552, (1, 0): 0.9561896472033945, (2, 0): 0.7396943865284514,
                 (3, 0): 1.2471153535887813, (0, 1): 0.5112281013261678, (1, 1): 1.4889814846642655,
                 (0, 2): 1.2734034136130559, (1, 2): 1.3582928186375915, (2, 2): 1.3712849069142987,
                 (3, 2): 1.0595326619190617, (4, 2): 1.2476590859827423, (0, 3): 0.7434610967377496,
                 (1, 3): 1.0235304695281342, (2, 3): 0.5799444598510225, (0, 4): 1.0148415054695707, (1, 4): 0,
                 (4, 0): 1.1260343848045027, (2, 1): 1.3473132221162576, (5, 2): 0.713117028158188,
                 (3, 3): 0.9567324181520453, (2, 4): 0.7530313621772257, 'depot': 1.3764411597645112},
        (4, 0): {(0, 0): 1.3664485054342281, (1, 0): 0.5408136956009475, (2, 0): 0.9992054954202398,
                 (3, 0): 0.9366736932941748, (0, 1): 0.7314755572690489, (1, 1): 1.2734651704291096,
                 (0, 2): 1.3543038982710411, (1, 2): 0.768574473628665, (2, 2): 1.3487910079665797,
                 (3, 2): 1.1470079358790768, (4, 2): 1.2529846548328383, (0, 3): 1.2924563210727702,
                 (1, 3): 0.7650314351712586, (2, 3): 1.1411733942868383, (0, 4): 1.3343907476280226,
                 (1, 4): 0.7379039814678413, 'depot': 1.0643089631415388, (4, 0): 0, (2, 1): 1.0782730926208628,
                 (5, 2): 1.2105855197389603, (3, 3): 1.374501595616925, (2, 4): 1.3241942990650264},
        (2, 1): {(0, 0): 1.04119884867173, (1, 0): 1.076463073940626, (2, 0): 1.1210088603013755,
                 (3, 0): 1.2409760279440818, (0, 1): 0.5486814364731731, (1, 1): 1.0404336019543887,
                 (0, 2): 0.7947214120642437, (1, 2): 1.027303374918744, (2, 2): 0.8210861553280253,
                 (3, 2): 1.3636967400526183, (4, 2): 1.4569948793396237, (0, 3): 1.1965798646724557,
                 (1, 3): 1.3203236491588952, (2, 3): 1.0745603887344166, (0, 4): 1.4378230249685793,
                 (1, 4): 1.1034550765027067, 'depot': 1.0421842778040367, (4, 0): 1.1460148986745171, (2, 1): 0,
                 (5, 2): 1.154414774468217, (3, 3): 0.8442891644661848, (2, 4): 0.6775079765034714},
        (5, 2): {(0, 0): 1.3806439413028748, (1, 0): 1.2811904310836204, (2, 0): 1.379992781291299,
                 (3, 0): 1.4876418336022232, (0, 1): 0.7872641203623587, (1, 1): 0.9860136687271156,
                 (0, 2): 1.039675576109473, (1, 2): 1.4928517821780556, (2, 2): 1.0304766182699323,
                 (3, 2): 1.4915535958349502, (4, 2): 0.6824907366785071, (0, 3): 0.9872249989623492,
                 (1, 3): 1.1834279422420138, (2, 3): 1.2442335380690426, (0, 4): 0.5377348213585114,
                 (1, 4): 1.4342604725363493, 'depot': 0.5324863281092008, (4, 0): 1.3338688503829395,
                 (2, 1): 1.4767188243880436, (5, 2): 0, (3, 3): 0.6619417250959763, (2, 4): 1.2659501195117242},
        (3, 3): {(0, 0): 0.7510898428319387, (1, 0): 0.706644033284897, (2, 0): 1.4533800872713494,
                 (3, 0): 1.3673538181309497, (0, 1): 1.1958113153935328, (1, 1): 0.8243128256368144,
                 (0, 2): 0.6277521366061564, (1, 2): 0.7686829544231357, (2, 2): 0.9717198494792566,
                 (3, 2): 0.5277559436123048, (4, 2): 1.0817139377499712, (0, 3): 0.6608486288716443,
                 (1, 3): 0.6543507477313877, (2, 3): 0.5955979375558281, (0, 4): 1.195723882469744,
                 (1, 4): 0.6533204587985224, 'depot': 1.4894295144338754, (4, 0): 0.7424910783707455,
                 (2, 1): 1.3649908348434159, (5, 2): 1.4023921635637038, (3, 3): 0, (2, 4): 0.6053715968649989},
        (2, 4): {(0, 0): 0.9890168440219794, (1, 0): 1.0707689857845208, (2, 0): 1.0448533075595638,
                 (3, 0): 1.3781286772453987, (0, 1): 1.027330708569604, (1, 1): 1.0004748078864363,
                 (0, 2): 0.502275079802843, (1, 2): 1.4473721137265443, (2, 2): 0.949122029077772,
                 (3, 2): 0.8394222390165847, (4, 2): 1.078956129158605, (0, 3): 0.8759612384882739,
                 (1, 3): 0.592145671474956, (2, 3): 1.3987584170531893, (0, 4): 0.8269326561177565,
                 (1, 4): 1.0307504779353829, 'depot': 0.7862586795907547, (4, 0): 0.5742672280971953,
                 (2, 1): 0.5241985488442352, (5, 2): 0.7953012393074342, (3, 3): 1.045625397728374, (2, 4): 0}}

    if AGV_selection == 0:
        travel_time_matrix_static = {'depot': {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (0, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (1, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (2, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (3, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (0, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (1, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (0, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (1, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (2, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (3, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (4, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (0, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (1, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (2, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (0, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (1, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0}, (4, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}, (2, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}, (5, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}, (3, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}, (2, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0, (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}}
        agvsPerWC = [6, 6, 6, 6, 6]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
                         [25, 26, 27, 28, 29, 30]]  # Index of AGVs

    # AGV information (6 AGV per WC)
    if AGV_selection == 1:

        agvsPerWC = [6, 6, 6, 6, 6]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
                         [25, 26, 27, 28, 29, 30]]  # Index of AGVs

    # AGV information (3 AGV per WC)
    if AGV_selection == 2:

        agvsPerWC = [3, 3, 3, 3, 3]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]  # Index of AGVs

    # AGV information (No_AGVS = Machines + 1)
    if AGV_selection == 3:

        agvsPerWC = [5, 3, 6, 4, 3]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18],
                         [19, 20, 21]]  # Index of AGVs

    # AGV information (No_AGVS = Machines + 2)
    if AGV_selection == 4:

        agvsPerWC = [6, 4, 7, 5, 4]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22],
                         [23, 24, 25, 26]]  # Index of AGVs

    # AGV information (No_AGVS = Machines - 1)
    if AGV_selection == 5:

        agvsPerWC = [3, 1, 4, 2, 1]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3], [4], [5, 6, 7, 8], [9, 10],
                         [11]]  # Index of AGVs

    # AGV information (No_AGVS = Machines)
    if AGV_selection == 6:

        agvsPerWC = [4, 2, 5, 3, 2]  # Number of AGVs per workcenter
        agv_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]  # Index of AGVs

    if AGV_selection == 7:

        agvsPerWC = agvsPerWC_input  # Number of AGVs per workcenter
        agv_number_WC = []
        agv_count = 1
        for agv_WC in agvsPerWC:
            WC_AGV_list = []
            for No_agv_WC in range(agv_WC):
                WC_AGV_list.append(No_agv_WC+agv_count)
            agv_count += agv_WC
            agv_number_WC.append(WC_AGV_list)

    return travel_time_matrix_static, agvsPerWC, agv_number_WC


def create_distance_matrix():
    """Creates distance matrix where distance can be requested by inputting:
    distance_maxtrix[actual location][destination]"""

    # All distances are in meters
    distance_matrix = {
        "depot": {(ii, jj): random.uniform(0.5, 1.5) for jj in noOfWC for ii in range(machinesPerWC[jj])}}

    distance_matrix["depot"].update(
        {(ii + machinesPerWC[jj], jj): random.uniform(0.5, 1.5) for jj in noOfWC for ii in range(noOfCbPerWC[jj])})

    distance_matrix["depot"].update({"depot": 0})

    # TODO: Random dinstance_matrix has to be placed before the simpy enviroment as a static dictionary

    for jj in noOfWC:
        for ii in range(machinesPerWC[jj]):
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0.5, 1.5) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update(
                {(ii + machinesPerWC[jj], jj): random.uniform(0.5, 1.5) for jj in noOfWC for ii in range(noOfCbPerWC[jj])})

            distance_matrix[(ii, jj)].update({"depot": random.uniform(0.5, 1.5)})

            distance_matrix[ii, jj][ii, jj] = 0

    for jj in noOfWC:
        for ii in range(noOfCbPerWC[jj]):
            ii += machinesPerWC[jj]
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0.5, 1.5) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update({"depot": random.uniform(0.5, 1.5)})

            distance_matrix[(ii, jj)].update({(ii + machinesPerWC[jj], jj): random.uniform(0.5, 1.5) for jj in
                                              noOfWC for ii in range(noOfCbPerWC[jj])})

            distance_matrix[ii, jj][ii, jj] = 0

    return distance_matrix

