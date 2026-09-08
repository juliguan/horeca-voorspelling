#!/usr/bin/env python3
"""Zelfstandige synthetische kassagenerator; Python 3.10+, numpy en pandas.
Gebruik: python kantoorlunchroom.py [--seed INTEGER] [--output PAD.csv]
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
PREFIX = 'KL'
DEFAULT_SEED = 582341
MENU = [('Koffie', 'warm', 285), ('Cappuccino', 'warm', 330), ('Thee', 'warm', 270), ('Water', 'fris', 250), ('Jus', 'fris', 425), ('IJsthee', 'fris', 310), ('Broodje kaas', 'gerecht', 625), ('Broodje kip', 'gerecht', 795), ('Broodje hummus', 'gerecht', 725), ('Maaltijdsalade', 'gerecht', 1050), ('Tomatensoep', 'soep', 550), ('Dagsoep', 'soep', 595), ('Banaan', 'extra', 150), ('Yoghurt', 'extra', 375), ('Muffin', 'extra', 325)]
BASE_GROUPS = [[5, 1, 0.8, 0.2, 2], [1.8, 2, 6, 1.6, 1.3], [3, 1, 1, 0.4, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
GUEST_P = [0.57, 0.28, 0.1, 0.035, 0.01, 0.005]
CAPACITY = np.array([65, 200, 35, 0, 0])  # gasten per dagdeel, inclusief tafelomloop


def overdispersed(rng, mean, shape):
    return int(rng.poisson(rng.gamma(shape, max(float(mean), 0.0) / shape)))


def season(d):
    # Jaarfase op maand/dag, zodat schrikkeljaren geen fasebreuk geven.
    return np.cos(2 * np.pi * (d.dayofyear - 205) / (366 if d.is_leap_year else 365))


TIME_CENTERS = [8.8, 12.25, 14.7, 18, 22]
TIME_WIDTHS = [0.55, 0.34, 0.35, 0.5, 0.5]
MEAL_CHANCE = [0.3, 0.97, 0.4, 0.1, 0.1]
LINE_P = [0.43, 0.39, 0.15, 0.03]
ROUND_CHANCE = 0
PRICE_CHANCE = 0.14
GAP_DAYS = 7
GAP_BLOCK = 3

def demand(dates, rng):
    result = []
    presence = .0
    for i,d in enumerate(dates):
        presence = .7 * presence + rng.normal(0,.065)
        weekday = [0.74,1.12,1.04,1.15,.60,0,0][d.weekday()]
        holiday = (d.month == 8 or (d.month == 7 and d.day >= 15))
        year_end = d.month == 12 and d.day >= 23 or d.month == 1 and d.day <= 2
        # Fictieve aangrenzende kantoorhuurders met afzonderlijke veranderingen.
        tenants = 1 if i < 270 else (.80 if i < 515 else 1.13)
        workers = int(max(0, 285 * weekday * tenants * (0.43 if holiday else 1) * np.exp(presence)))
        training = d.weekday() in (1,3) and rng.random() < .08
        hungry = rng.binomial(workers, rng.beta(16,40))
        lunch_orders = int(round(hungry / 1.28)) + (int(rng.integers(9,19)) if training else 0)
        orders = [overdispersed(rng,workers*.045,18), lunch_orders,
                  overdispersed(rng,workers*.016,7),0,0]
        if d.weekday() >= 5 or year_end or (d.month,d.day) in [(4,27),(12,25),(12,26),(1,1)]:
            orders = [0]*5
        # Zeldzame besloten zaterdaglunch; zondag altijd gesloten.
        if d.weekday() == 5 and not year_end and rng.random() < .025:
            orders = [0,int(rng.integers(8,18)),0,0,0]
        result.append(dict(orders=orders, training=training, time_shift=-.12 if training else 0))
    return result


def adjust_basket(w, part, ctx):
    if ctx['training']:
        w[0] *= 1.35
        w[4] *= 1.2
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
