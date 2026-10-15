"""Pure eligibility decisions for the matched corrected conditioning suite."""
COMPOUND_MEMBERS = {'Lisa+Annie': ('Lisa', 'Annie')}


def recording_group(row):
    # Creator confirms distinct compositions; these roles reuse source audio.
    song, part = row['song_id'], row['part']
    if song in ('O008', 'O009') and part in ('A', 'B', 'VP'):
        return 'shared_O008_O009:' + part
    return song + ':' + part


def legal_recording_enrollment(target, candidate):
    return (target['singer_id'] == candidate['singer_id']
            and target['song_id'] != candidate['song_id']
            and recording_group(target) != recording_group(candidate)
            and target['source_audio_sha256'] != candidate['source_audio_sha256'])


def members(identity):
    if identity in ('', '0', 'None'):
        return ()
    if '+' in identity:
        if identity not in COMPOUND_MEMBERS:
            raise ValueError('Unverified compound singer identity: ' + identity)
        return COMPOUND_MEMBERS[identity]
    return (identity,)


def conflicting_roles(target, sources, active_starts):
    start = float(target['target_start'])
    conflicts = []
    for source in sources:
        if source['part'] == target['part']:
            continue
        if start not in active_starts[source['stem_path']]:
            continue
        identities = members(source['singer_id'])
        if not identities:
            raise ValueError('Active source with unknown identity: ' + source['stem_path'])
        if target['singer_id'] in identities:
            conflicts.append(source['part'])
    return sorted(set(conflicts))


def legal_gallery(target, gallery_compositions):
    return bool(gallery_compositions.get(target['singer_id'], set()) - {target['song_id']})
