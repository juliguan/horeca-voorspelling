#!/usr/bin/env python3
"""Zelfstandige synthetische kassagenerator; Python 3.10+, numpy en pandas.
Gebruik: python grandcafe.py [--seed INTEGER] [--output PAD.csv]
Geen netwerk, externe bestanden of imports uit de andere generatoren.
Alle bedragen zijn EUR inclusief btw; synthetische aannames, geen gemeten data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['order_id', 'tijdstip', 'dagdeel', 'aantal_gasten', 'item',
           'categorie', 'aantal', 'stukprijs', 'regelbedrag']
PARTS = ['ochtend', 'lunch', 'middag', 'diner', 'avond']
# Lokale, tijdzone-naieve kalenderdatum; geen boekingen over middernacht.
WINDOWS = [(6, 11), (11, 14), (14, 17), (17, 21), (21, 24)]
PREFIX = 'GC'
DEFAULT_SEED = 746129
MENU = [('Koffie', 'warm', 350), ('Cappuccino', 'warm', 395), ('Thee', 'warm', 340), ('Jus', 'fris', 495), ('Water', 'fris', 350), ('Limonade', 'fris', 425), ('Bier', 'alcohol', 450), ('Huiswijn', 'alcohol', 650), ('Spritz', 'alcohol', 1095), ('Croissant', 'gebak', 350), ('Appeltaart', 'gebak', 575), ('Brownie', 'gebak', 495), ('Uitsmijter', 'gerecht', 1350), ('Clubsandwich', 'gerecht', 1650), ('Burger', 'gerecht', 2050), ('Salade', 'gerecht', 1850), ('Bitterballen', 'snack', 925), ('Borrelplank', 'snack', 1850)]
BASE_GROUPS = [[4, 2, 0, 4, 1, 0.1], [1, 2, 1, 0.5, 6, 0.5], [3, 3, 1.4, 4, 1, 1], [0.7, 2, 3, 0.2, 6, 1], [0.4, 1, 6, 0.2, 0.7, 3]]
GUEST_P = [0.18, 0.4, 0.2, 0.14, 0.06, 0.02]
CAPACITY = np.array([80, 160, 190, 150, 110])  # gasten per dagdeel, inclusief tafelomloop


def overdispersed(rng, mean, shape):
    return int(rng.poisson(rng.gamma(shape, max(float(mean), 0.0) / shape)))


def season(d):
    # Jaarfase op maand/dag, zodat schrikkeljaren geen fasebreuk geven.
    return np.cos(2 * np.pi * (d.dayofyear - 205) / (366 if d.is_leap_year else 365))


TIME_CENTERS = [9.7, 12.8, 15.2, 18.6, 22]
TIME_WIDTHS = [0.7, 0.7, 0.8, 0.9, 0.7]
MEAL_CHANCE = [0.75, 0.9, 0.75, 0.9, 0.5]
LINE_P = [0.2, 0.38, 0.29, 0.13]
ROUND_CHANCE = 0.24
PRICE_CHANCE = 0.22
GAP_DAYS = 4
GAP_BLOCK = 2

def demand(dates, rng):
    # Volledig fictieve evenementen; verschillende duur en bezoekersstromen.
    events = np.zeros(len(dates))
    for year in (2023,2024,2025,2026):
        positions = [i for i,d in enumerate(dates) if d.year == year]
        for start in rng.choice(positions,size=min(11,len(positions)),replace=False):
            length = int(rng.integers(1,6))
            amplitude = rng.uniform(.25,1.35)
            for j in range(start,min(start+length,len(dates))):
                events[j] += amplitude * (1 - .12*(j-start))
    result=[]
    travel=.0
    for i,d in enumerate(dates):
        travel = .94*travel + rng.normal(0,.045)
        summer = np.exp(.43*season(d))
        festive = 1.45 if d.month==12 and 8<=d.day<=30 else 1
        spring = 1.25 if d.month in (4,5) else 1
        traffic = 70*summer*festive*spring*np.exp(travel)
        traffic *= 1+events[i]
        traffic *= [1,.96,1.02,1.04,1.10,1.17,1.12][d.weekday()]
        # Reizigersgroepen geven incidentele scheve pieken, met beperkte capaciteit.
        bus = int(rng.poisson(2)) if rng.random()<.11 else 0
        diversion = .35 if rng.random()<.008 else 1
        base=np.array([.16,.30,.36,.25,.12])*traffic*diversion
        base[1:3] += bus*9
        if events[i] > .65:
            base[3:5] *= 1.5
        orders=[overdispersed(rng,m,22) for m in base]
        if d.month==1 and d.day in (1,2):
            orders=[0]*5
        result.append(dict(orders=orders,event=events[i],summer=summer,bus=bus))
    return result


def adjust_basket(w, part, ctx):
    if ctx['summer'] > 1.2:
        w[1] *= 1.5
        w[0] *= .75
    if ctx['event'] > .65 and part >= 3:
        w[2] *= 1.7
    if ctx['bus']:
        w[3] *= 1.4
    return w


def generate(seed=DEFAULT_SEED, output=None):
    # Gescheiden stromen: wijzigingen aan een mandje veranderen de vraag niet.
    streams = np.random.SeedSequence(seed).spawn(4)
    rng, rb, rp, rq = [np.random.default_rng(s) for s in streams]
    dates = pd.date_range('2023-09-01', '2026-08-31', freq='D')
    daily = demand(dates, rng)
    months = pd.period_range('2023-09', '2026-08', freq='M')
    # Geleidelijke, per product asynchrone prijsaanpassingen in hele centen.
    prices = {}
    for name, category, cents in MENU:
        current = cents
        path = []
        for m in range(len(months)):
            if m and rp.random() < PRICE_CHANCE:
                current = int(5 * round(current * (1 + rp.uniform(.006, .027)) / 5))
            path.append(current)
        prices[name] = path
    menu_groups = {}
    for product in MENU:
        menu_groups.setdefault(product[1], []).append(product)

    output = Path(output or Path(__file__).with_suffix('.csv'))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
    sequence = 0
    exported = 0
    # Exportstoringen laten alleen hele orders weg, nooit losse orderregels.
    gaps = set(rq.choice(np.arange(1, len(dates)-1), size=GAP_DAYS, replace=False).tolist())
    block_start = int(rq.integers(100, len(dates)-100))
    gaps.update(range(block_start, block_start + GAP_BLOCK))
    for i, (d, context) in enumerate(zip(dates, daily)):
        if i in gaps:
            continue
        partial = rq.random() < .018 and i not in (0, len(dates)-1)
        retention = rq.uniform(.35, .8) if partial else 1.0
        rows = []
        month = (d.year - 2023) * 12 + d.month - 9
        for part_idx, proposed in enumerate(context['orders']):
            guests_used = 0
            for _ in range(int(proposed)):
                guests = int(rb.choice(np.arange(1, 7), p=GUEST_P))
                if guests_used + guests > CAPACITY[part_idx]:
                    break
                guests_used += guests
                sequence += 1
                lo, hi = WINDOWS[part_idx]
                # Piek rond een dagspecifiek centrum, gemengd met inloop.
                center = TIME_CENTERS[part_idx] + context.get('time_shift', 0)
                hour = (rb.uniform(lo, hi) if rb.random() < .20 else
                        np.clip(rb.normal(center, TIME_WIDTHS[part_idx]), lo, hi - 1/3600))
                timestamp = d + pd.Timedelta(seconds=int(hour * 3600))
                weights = np.array(BASE_GROUPS[part_idx], dtype=float)
                weights = adjust_basket(weights, part_idx, context)
                weights /= weights.sum()
                group_names = list(menu_groups)
                # Eén mandje per bestelling, 2-5 verschillende producten.
                nlines = int(rb.choice([2, 3, 4, 5], p=LINE_P))
                selected = []
                # Eerste regel is steeds een passende drank; vervolgens vaak eten.
                drink_choices = [j for j, g in enumerate(group_names) if g in ('warm', 'fris', 'alcohol')]
                dw = weights[drink_choices]; dw /= dw.sum()
                first_group = group_names[int(rb.choice(drink_choices, p=dw))]
                selected.append(menu_groups[first_group][int(rb.integers(len(menu_groups[first_group])))])
                if rb.random() < MEAL_CHANCE[part_idx]:
                    food = [j for j,g in enumerate(group_names) if g not in ('warm','fris','alcohol') and weights[j] > 0]
                    fw = weights[food]; fw /= fw.sum()
                    g = group_names[int(rb.choice(food, p=fw))]
                    selected.append(menu_groups[g][int(rb.integers(len(menu_groups[g])))])
                while len(selected) < nlines:
                    available = [p for p in MENU if p not in selected and weights[group_names.index(p[1])] > 0]
                    pw = np.array([weights[group_names.index(p[1])] / len(menu_groups[p[1]]) for p in available])
                    pw /= pw.sum()
                    selected.append(available[int(rb.choice(len(available), p=pw))])
                keep = rq.random() < retention
                for name, category, _ in selected:
                    # Hoeveelheden delen de groepsgrootte; barbestellingen kunnen rondes bevatten.
                    q = 1 + int(rb.binomial(max(guests - 1, 0), .70 if category in ('warm','fris','alcohol') else .48))
                    if category == 'alcohol' and part_idx >= 3:
                        q += int(rb.binomial(guests, ROUND_CHANCE))
                    if category in ('snack','gebak','dessert'):
                        q = max(1, int(np.ceil(q / 2)))
                    cents = prices[name][month]
                    if keep:
                        rows.append((f'{PREFIX}-{sequence:09d}', timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                     PARTS[part_idx], guests, name, category, q,
                                     f'{cents / 100:.2f}', f'{q * cents / 100:.2f}'))
        if rows:
            frame = pd.DataFrame(rows, columns=COLUMNS)
            frame = frame.sort_values(['tijdstip', 'order_id'], kind='stable')
            frame.to_csv(output, mode='a', header=False, index=False)
            exported += len(frame)
    print(f'{output}: {exported:,} orderregels; seed={seed}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed moet niet-negatief zijn')
    generate(args.seed, args.output)


if __name__ == '__main__':
    main()
