"""Fixed constants of the KC map, its training law and the samplers.

Changing any value here changes what a checkpoint or an RNG stream means. Rates are in 1/ps, times in ps, lengths in
nm, energies in kBT unless stated otherwise.
"""

# Network gates G_k(rate * tau) = chi^k / (1 + chi^k), half-open at tau = 1/rate (3 fs).
POSITION_GATE_RATE_PS_INV = 1000.0 / 3.0  # position log-scale and shift gate
POSITION_GATE_ORDER = 3
MOMENTUM_LOG_SCALE_RATE_PS_INV = 1000.0 / 3.0  # momentum log-scale gate
MOMENTUM_LOG_SCALE_ORDER = 2
MOMENTUM_SHIFT_RATE_PS_INV = 1000.0 / 3.0  # momentum shift envelope
MOMENTUM_SHIFT_ORDER = 3

# Bounded heads: shifts pass through B tanh(b / B); log-scales through a tanh with these prefactors.
MOMENTUM_SHIFT_BOUND = 32.0
POSITION_SHIFT_BOUND = 4.0
POSITION_LOG_SCALE_PREFACTOR = 1.25
MOMENTUM_LOG_SCALE_BUDGET = 12.0  # shared over the num_blocks / 2 momentum blocks
POSITION_UNIT_LENGTH_NM = 0.010  # saturation length of the free-flight position-shift unit

# Mechanical clock s(h): identity through CLOCK_IDENTITY_PS, then saturating at CLOCK_IDENTITY_PS + CLOCK_TRANSITION_PS.
CLOCK_IDENTITY_PS = 7.5e-4
CLOCK_TRANSITION_PS = 7.5e-4
CONTEXT_LOG_WIDTH_FLOOR_PS = 1e-6  # smallest lag of the density and of the conditioner's width features
SOURCE_MIXING_TIME_FLOOR_PS = 1e-6  # smallest lag seen by the source mixer's rate features

# Conditioner input widths.
SPEED_BASIS = 8  # Gaussian bumps on the per-atom speed (thermal-momentum units)
SPEED_LIMIT = 6.0
IMPULSE_BASIS = 16  # Gaussian bumps on the per-atom mean force impulse (thermal-momentum units)
IMPULSE_LIMIT = 3.5
KINEMATIC_CHANNELS = SPEED_BASIS + IMPULSE_BASIS + 1  # plus the speed-impulse alignment cosine
TAU_FEATURES = 256  # sinusoidal lag features

# N-CA-C source frame validity.
SOURCE_FRAME_MIN_BOND_NM = 1e-6
SOURCE_FRAME_MIN_SIN = 1e-4

# Angular-momentum map T: rows with det T below this get zero training weight; the sampler redraws them.
ANGULAR_MOMENTUM_MAP_MIN_DET_RATIO = 0.05

# KC two-step target: the lag splits at a Beta(2, 2) fraction.
KC_SPLIT_BETA = 2.0

# Lane layout per block of rows on each device: LANE_BLOCK - SECOND_LANE_ROWS KC rows, then the DB (or PF) rows.
LANE_BLOCK = 4
SECOND_LANE_ROWS = 1
PF_ADANORM_EPS = 1e-3  # adaptive normalization of the PF residuals

# Numerics and RNG law of training.
MATMUL_PRECISION = "highest"
TRAINING_PRNG = {"implementation": "threefry2x32", "threefry_partitionable": True, "random_seed_offset": 0}
KC_SPLIT_FOLD_IN = 0x43534B53  # split-fraction stream of the KC target

# Adaptive-chains sampler streams, folded into a chain's seed key or a proposal's epsilon key.
CHAIN_MOMENTUM_STREAM = 0x50415254  # initial Maxwell momenta
CHAIN_STREAM = 0x43484149  # chain key
ENERGY_STREAM = 0x454E5247  # heat-bath draw of the energy filter
REDRAW_STREAM = 0x52445257  # redraw attempts n >= 1

# Metropolis-adjusted sampler streams.
MH_INITIAL_MOMENTUM_STREAM = 0
MH_PROPOSAL_NOISE_STREAM = 1
MH_ACCEPTANCE_STREAM = 2

# Energy filter: internal degrees of freedom are 3N - FILTER_DOF_OFFSET; |dU + dK_int| <= FILTER_MAX_DH_KBT
# sqrt((3N - FILTER_DOF_OFFSET) / FILTER_REFERENCE_DOF) and a kinetic target of at least FILTER_KINETIC_FLOOR_KBT.
FILTER_DOF_OFFSET = 6
FILTER_KINETIC_FLOOR_KBT = 0.1
FILTER_MAX_DH_KBT = 25.0
FILTER_REFERENCE_DOF = 60

# Sampler clock: 1/16 fs ticks; lags from LAG_MIN_TICKS (1.5625 fs) to LAG_MAX_TICKS (100 fs).
TICK_PS = 6.25e-5
TICKS_PER_FS = 16
TICKS_PER_PS = 1000 * TICKS_PER_FS
LAG_MIN_TICKS = 25
LAG_MAX_TICKS = 1600
