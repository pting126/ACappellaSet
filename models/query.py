"""Validate public queries without importing Torch."""
import math

ROLE_VOCAB = ('S', 'A', 'T', 'Bar', 'B', 'VP', 'Lead', 'None')
ALIASES = {'soprano': 'S', 'alto': 'A', 'tenor': 'T', 'baritone': 'Bar',
           'bass': 'B', 'vp': 'VP', 'lead': 'Lead', 'lead_vocal': 'Lead'}


def validate_query(spec, role=None, enrollment=None, enrollment_start=0.0,
                   allow_untrained_role=False):
    mode = spec['conditioning']
    if not math.isfinite(enrollment_start) or enrollment_start < 0:
        raise ValueError('Enrollment start must be finite and nonnegative')
    if mode == 'role':
        if enrollment is not None or enrollment_start != 0:
            raise ValueError('Part-conditioned models do not accept enrollment audio')
    elif enrollment is None:
        raise ValueError('Singer and dual models require --enrollment')
    if mode == 'enrollment':
        if role is not None or allow_untrained_role:
            raise ValueError('Singer-conditioned models use enrollment without a role query')
        return {'role': None, 'role_index': ROLE_VOCAB.index('None'),
                'mechanism': 'matching_target_enrollment'}
    if role is None:
        raise ValueError('Part and dual models require --role')
    canonical = role if role in ROLE_VOCAB else ALIASES.get(role.lower())
    if canonical is None or canonical == 'None':
        raise ValueError(f'Unknown vocal role: {role!r}')
    if canonical not in spec['training_roles']:
        if not allow_untrained_role or canonical != spec.get('untrained_zero_role'):
            raise ValueError(f'{canonical} was not supervised for this model; '
                             'the paper zero-token diagnostic requires --allow-untrained-role')
        mechanism = 'existing_zero_role' if mode == 'role' else 'enrollment_plus_existing_zero_role'
    else:
        mechanism = 'trained_role_token' if mode == 'role' else 'matching_role_and_enrollment'
    return {'role': canonical, 'role_index': ROLE_VOCAB.index(canonical), 'mechanism': mechanism}
