#!/usr/bin/env python3
"""Ingest the clean NBA player snapshot and Reddit OCR documents into PostgreSQL."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import (CheckConstraint, Column, Date, DateTime, ForeignKey, Index,
                        Integer, MetaData, Numeric, SmallInteger, String, Table, Text,
                        UniqueConstraint, create_engine, func, text)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

HERE = Path(__file__).resolve().parent
DEFAULT_XLSX = Path('/mnt/user-data/uploads/regular NBA.xlsx')
OCR_FILES = [Path(f'/mnt/user-data/uploads/Reddit {n}.pdf.ocr.md') for n in range(1, 5)]
DATASET_KEY = 'regular-nba-donnees-nba-player-team-snapshot-v1'

# Correspondance explicite entre en-têtes source et noms SQL/Pydantic stables.
NUMERIC_MAP = {
    'Age':'age','GP':'gp','W':'wins','L':'losses','Min':'minutes','PTS':'points',
    'FGM':'field_goals_made','FGA':'field_goals_attempted','FG%':'field_goal_pct',
    'metric_15':'metric_15_00','3PA':'three_pointers_attempted','3P%':'three_point_pct',
    'FTM':'free_throws_made','FTA':'free_throws_attempted','FT%':'free_throw_pct',
    'OREB':'offensive_rebounds','DREB':'defensive_rebounds','REB':'rebounds','AST':'assists',
    'TOV':'turnovers','STL':'steals','BLK':'blocks','PF':'personal_fouls','FP':'fantasy_points',
    'DD2':'double_doubles','TD3':'triple_doubles','+/-':'plus_minus','OFFRTG':'off_rating',
    'DEFRTG':'def_rating','NETRTG':'net_rating','AST%':'assist_pct','AST/TO':'assist_to_turnover',
    'AST RATIO':'assist_ratio','OREB%':'offensive_rebound_pct','DREB%':'defensive_rebound_pct',
    'REB%':'rebound_pct','TO RATIO':'turnover_ratio','EFG%':'effective_fg_pct','TS%':'true_shooting_pct',
    'USG%':'usage_pct','PACE':'pace','PIE':'pie','POSS':'possessions',
}
INTEGER_FIELDS = {'age','gp','wins','losses','points','field_goals_made','field_goals_attempted',
 'metric_15_00','three_pointers_attempted','free_throws_made','free_throws_attempted',
 'offensive_rebounds','defensive_rebounds','rebounds','assists','turnovers','steals','blocks',
 'personal_fouls','fantasy_points','double_doubles','triple_doubles','possessions'}

class PlayerRow(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    source_name: str = Field(min_length=1, max_length=300)

    @field_validator('source_name')
    @classmethod
    def nonblank(cls, v: str) -> str:
        if not v.strip(): raise ValueError('nom joueur vide')
        return v

class StatsRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    team_code: str = Field(pattern=r'^[A-Z]{3}$')
    age: int = Field(ge=0, le=120)
    gp: int = Field(ge=0, le=82)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    minutes: float
    points: int = Field(ge=0)
    field_goals_made: int = Field(ge=0)
    field_goals_attempted: int = Field(ge=0)
    field_goal_pct: float
    metric_15_00: int
    three_pointers_attempted: int = Field(ge=0)
    three_point_pct: float
    free_throws_made: int = Field(ge=0)
    free_throws_attempted: int = Field(ge=0)
    free_throw_pct: float
    offensive_rebounds: int = Field(ge=0)
    defensive_rebounds: int = Field(ge=0)
    rebounds: int = Field(ge=0)
    assists: int = Field(ge=0)
    turnovers: int = Field(ge=0)
    steals: int = Field(ge=0)
    blocks: int = Field(ge=0)
    personal_fouls: int = Field(ge=0)
    fantasy_points: int
    double_doubles: int = Field(ge=0)
    triple_doubles: int = Field(ge=0)
    plus_minus: float
    off_rating: float
    def_rating: float
    net_rating: float
    assist_pct: float
    assist_to_turnover: float
    assist_ratio: float
    offensive_rebound_pct: float
    defensive_rebound_pct: float
    rebound_pct: float
    turnover_ratio: float
    effective_fg_pct: float
    true_shooting_pct: float
    usage_pct: float
    pace: float
    pie: float
    possessions: int = Field(ge=0)

    @field_validator('team_code')
    @classmethod
    def strip_team(cls, v: str) -> str: return v.strip().upper()

    @field_validator('minutes','field_goal_pct','three_point_pct','free_throw_pct','off_rating','def_rating','net_rating','assist_pct','assist_to_turnover','assist_ratio','offensive_rebound_pct','defensive_rebound_pct','rebound_pct','turnover_ratio','effective_fg_pct','true_shooting_pct','usage_pct','pace','pie','plus_minus')
    @classmethod
    def finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v): raise ValueError('valeur non finie')
        return v

    @field_validator('gp')
    @classmethod
    def validate_win_loss(cls, v: int, info):
        # Le contrôle croisé est fait après validation du modèle via model_validator ci-dessous.
        return v

    @classmethod
    def from_source(cls, row: dict[str, Any]) -> 'StatsRow':
        data={'team_code':row['Team']}
        for source, target in NUMERIC_MAP.items():
            value=row[source]
            if pd.isna(value): raise ValueError(f'{source}: valeur manquante')
            if target in INTEGER_FIELDS:
                fv=float(value)
                if not fv.is_integer(): raise ValueError(f'{source}: entier attendu, reçu {value!r}')
                value=int(fv)
            else: value=float(value)
            data[target]=value
        obj=cls.model_validate(data)
        if obj.wins + obj.losses != obj.gp:
            raise ValueError(f'W + L != GP ({obj.wins} + {obj.losses} != {obj.gp})')
        return obj

# Core metadata is also used for SQLite validation and mirrors schema.sql.
metadata=MetaData()
teams=Table('teams',metadata,Column('team_code',String(3),primary_key=True),Column('team_name',Text,nullable=False,unique=True))
players=Table('players',metadata,Column('player_id',Integer,primary_key=True,autoincrement=True),Column('source_name',Text,nullable=False,unique=True),Column('created_at',DateTime(timezone=True),server_default=func.now(),nullable=False))
stats_columns=[Column('stats_id',Integer,primary_key=True,autoincrement=True),Column('player_id',Integer,ForeignKey('players.player_id',ondelete='CASCADE'),nullable=False),Column('team_code',String(3),ForeignKey('teams.team_code'),nullable=False),Column('season_label',Text),Column('dataset_key',Text,nullable=False)]
for key in ['age','gp','wins','losses']: stats_columns.append(Column(key,SmallInteger,nullable=False))
stats_columns += [Column('minutes',Numeric(12,4),nullable=False)]
for key in ['points','field_goals_made','field_goals_attempted']: stats_columns.append(Column(key,Integer,nullable=False))
stats_columns += [Column('field_goal_pct',Numeric(12,4),nullable=False),Column('metric_15_00',Integer,nullable=False),Column('three_pointers_attempted',Integer,nullable=False),Column('three_point_pct',Numeric(12,4),nullable=False),Column('free_throws_made',Integer,nullable=False),Column('free_throws_attempted',Integer,nullable=False),Column('free_throw_pct',Numeric(12,4),nullable=False)]
for key in ['offensive_rebounds','defensive_rebounds','rebounds','assists','turnovers','steals','blocks','personal_fouls']: stats_columns.append(Column(key,Integer,nullable=False))
stats_columns += [Column('fantasy_points',Integer,nullable=False),Column('double_doubles',Integer,nullable=False),Column('triple_doubles',Integer,nullable=False),Column('plus_minus',Numeric(12,4),nullable=False)]
for key in ['off_rating','def_rating','net_rating','assist_pct','assist_to_turnover','assist_ratio','offensive_rebound_pct','defensive_rebound_pct','rebound_pct','turnover_ratio','effective_fg_pct','true_shooting_pct','usage_pct','pace','pie']: stats_columns.append(Column(key,Numeric(12,4),nullable=False))
stats_columns += [Column('possessions',Integer,nullable=False),Column('source_file',Text,nullable=False),Column('source_sheet',Text,nullable=False),Column('source_row',Integer,nullable=False),Column('loaded_at',DateTime(timezone=True),server_default=func.now(),nullable=False)]
stats=Table('stats',metadata,*stats_columns,UniqueConstraint('dataset_key','player_id','team_code',name='uq_stats_dataset_player_team'),CheckConstraint('wins + losses = gp',name='ck_stats_win_loss_games'))
matches=Table('matches',metadata,Column('match_id',Integer,primary_key=True,autoincrement=True),Column('match_date',Date,nullable=False),Column('season_label',Text),Column('home_team_code',String(3),ForeignKey('teams.team_code'),nullable=False),Column('away_team_code',String(3),ForeignKey('teams.team_code'),nullable=False),Column('home_score',SmallInteger),Column('away_score',SmallInteger),Column('source_ref',Text),CheckConstraint('home_team_code <> away_team_code',name='ck_matches_distinct_teams'),UniqueConstraint('match_date','home_team_code','away_team_code',name='uq_matches_fixture'))
reports=Table('reports',metadata,Column('report_id',Integer,primary_key=True,autoincrement=True),Column('source_platform',Text,nullable=False,server_default='Reddit'),Column('subreddit',Text,nullable=False),Column('post_id',Text,nullable=False),Column('source_url',Text,nullable=False,unique=True),Column('author',Text),Column('title',Text,nullable=False),Column('body_text',Text,nullable=False),Column('raw_ocr_text',Text,nullable=False),Column('capture_date_text',Text),Column('source_file',Text,nullable=False),Column('created_at',DateTime(timezone=True),server_default=func.now(),nullable=False),UniqueConstraint('source_platform','post_id',name='uq_reports_platform_post'))
Index('ix_stats_player',stats.c.player_id); Index('ix_stats_team',stats.c.team_code); Index('ix_stats_season',stats.c.season_label); Index('ix_matches_date',matches.c.match_date); Index('ix_matches_season',matches.c.season_label); Index('ix_reports_subreddit',reports.c.subreddit)

REPORT_META=[
('1k9wrcj','MannerSuperb','Who are teams in the playoffs that have impressed you ?','1.pdf'),
('1kxixih','mokaloca82','How is it that the two best teams in the playoffs based on stats, having a chance of playing against each other in the Finals, is considered to be a snoozefest?','2.pdf'),
('11161bp','hqppp','Reggie Miller is the most efficient first option in NBA playoff history','3.pdf'),
('1l066qq','DonT012','Which NBA team did not have home court advantage until the NBA Finals?','4.pdf'),
]
def read_excel(path: Path) -> pd.DataFrame:
    frame=pd.read_excel(path,sheet_name='Données NBA',header=1,usecols='A:AS')
    # pandas peut convertir l'en-tête source 15:00 en datetime.time(15,0).
    renamed={c:('metric_15' if isinstance(c,time) and c.hour==15 and c.minute==0 else c) for c in frame.columns}
    frame=frame.rename(columns=renamed)
    expected=['Player','Team',*NUMERIC_MAP.keys()]
    if list(frame.columns)!=expected:
        raise ValueError('En-têtes Excel inattendus: '+repr(list(frame.columns)))
    return frame

def extract_body(raw: str, title: str) -> str:
    lines=raw.splitlines()
    start=next((i for i,line in enumerate(lines) if line.strip().lstrip('#').strip()==title),None)
    if start is None: return ''
    chunks=[]
    url=re.compile(r'https?://')
    for line in lines[start+1:]:
        s=line.strip()
        if url.search(s) or s.startswith('## Page number') or s.startswith('### Images found'): break
        if not s or s.startswith('#') or s in {'Accéder au contenu principal','Se connecter','Rechercher dans r/nba','Partager'}: continue
        # Voter/compteur Reddit isolé : marque la fin du texte original initial.
        if re.fullmatch(r'[\d,.kKmM]+',s) and chunks: break
        if 'Sponsorisé' in s: break
        chunks.append(s)
    return '\n\n'.join(chunks).strip()

def ingest_reports(conn, include: bool):
    if not include: return 0
    inserted=0
    for (post_id,author,title,filename),path in zip(REPORT_META,OCR_FILES):
        raw=path.read_text(encoding='utf-8')
        match=re.search(r'https?://www\.reddit\.com/r/nba/comments/'+re.escape(post_id)+r'/[^\s]+',raw)
        url=match.group(0).rstrip('.,') if match else f'https://www.reddit.com/r/nba/comments/{post_id}/'
        capture=next((ln.strip() for ln in raw.splitlines() if re.match(r'^\d{2}/\d{2}/\d{4}',ln.strip())),None)
        values={'source_platform':'Reddit','subreddit':'nba','post_id':post_id,'source_url':url,'author':author,'title':title,'body_text':extract_body(raw,title),'raw_ocr_text':raw,'capture_date_text':capture,'source_file':f'Reddit {filename}.ocr.md'}
        stmt=sqlite_insert(reports).values(**values).on_conflict_do_update(index_elements=['source_platform','post_id'],set_={k:values[k] for k in ['source_url','author','title','body_text','raw_ocr_text','capture_date_text','source_file']}) if conn.dialect.name=='sqlite' else pg_insert(reports).values(**values).on_conflict_do_update(index_elements=['source_platform','post_id'],set_={k:values[k] for k in ['source_url','author','title','body_text','raw_ocr_text','capture_date_text','source_file']})
        conn.execute(stmt); inserted+=1
    return inserted

def run(args):
    url=args.database_url or os.getenv('DATABASE_URL')
    if not url: raise ValueError('Définir DATABASE_URL (ex. postgresql+psycopg2://...) ou utiliser --database-url.')
    if url.startswith('postgres://'): url='postgresql+psycopg2://'+url[len('postgres://'):]
    if url.startswith('postgresql://'): url='postgresql+psycopg2://'+url[len('postgresql://'):]
    engine=create_engine(url)
    frame=read_excel(Path(args.xlsx))
    team_ref=pd.read_excel(Path(args.xlsx),sheet_name='Equipe')
    team_names={str(r['Code']).strip().upper():str(r["Nom complet de l'équipe"]).strip() for _,r in team_ref.iterrows()}
    rejects=[]; valid=[]; seen=set()
    for idx,row in frame.iterrows():
        excel_row=int(idx)+3
        try:
            p=PlayerRow.model_validate({'source_name':row['Player']})
            s=StatsRow.from_source(row.to_dict())
            if s.team_code not in team_names: raise ValueError(f'équipe absente du référentiel: {s.team_code}')
            key=p.source_name.casefold()
            if key in seen: raise ValueError('nom de joueur dupliqué dans la plage source')
            seen.add(key)
            valid.append((p,s,excel_row))
        except (ValidationError,ValueError,TypeError) as exc:
            rejects.append({'excel_row':excel_row,'player':str(row.get('Player','')),'error':str(exc)})
    if not valid: raise ValueError('Aucune ligne valide à ingérer.')
    with engine.begin() as conn:
        if engine.dialect.name == 'postgresql':
            ddl=(HERE/'schema.sql').read_text(encoding='utf-8')
            for statement in ddl.split(';'):
                if statement.strip(): conn.exec_driver_sql(statement)
        metadata.create_all(conn)
        # Référentiel contrôlé dédié, codes liés via FK aux stats.
        for code,name in team_names.items():
            stmt=sqlite_insert(teams).values(team_code=code,team_name=name).on_conflict_do_update(index_elements=['team_code'],set_={'team_name':name}) if conn.dialect.name=='sqlite' else pg_insert(teams).values(team_code=code,team_name=name).on_conflict_do_update(index_elements=['team_code'],set_={'team_name':name})
            conn.execute(stmt)
        player_ids={}
        for p,_,_ in valid:
            stmt=sqlite_insert(players).values(source_name=p.source_name).on_conflict_do_nothing(index_elements=['source_name']) if conn.dialect.name=='sqlite' else pg_insert(players).values(source_name=p.source_name).on_conflict_do_nothing(index_elements=['source_name'])
            conn.execute(stmt)
        for p,_,_ in valid:
            player_ids[p.source_name]=conn.execute(text('SELECT player_id FROM players WHERE source_name=:n'),{'n':p.source_name}).scalar_one()
        for p,s,rn in valid:
            payload=s.model_dump()
            payload.update(player_id=player_ids[p.source_name],season_label=None,dataset_key=DATASET_KEY,source_file=Path(args.xlsx).name,source_sheet='Données NBA',source_row=rn)
            stmt=sqlite_insert(stats).values(**payload).on_conflict_do_update(index_elements=['dataset_key','player_id','team_code'],set_={k:v for k,v in payload.items() if k not in {'dataset_key','player_id','team_code'}}) if conn.dialect.name=='sqlite' else pg_insert(stats).values(**payload).on_conflict_do_update(index_elements=['dataset_key','player_id','team_code'],set_={k:v for k,v in payload.items() if k not in {'dataset_key','player_id','team_code'}})
            conn.execute(stmt)
        reports_n=ingest_reports(conn,not args.skip_reports)
        players_n=conn.execute(text('SELECT count(*) FROM players')).scalar_one()
        stats_n=conn.execute(text('SELECT count(*) FROM stats WHERE dataset_key=:d'),{'d':DATASET_KEY}).scalar_one()
        teams_n=conn.execute(text('SELECT count(*) FROM teams')).scalar_one()
        matches_n=conn.execute(text('SELECT count(*) FROM matches')).scalar_one()
        # Tests de requêtes métier réelles : leaders en points et moyennes par équipe.
        leaders=conn.execute(text('SELECT p.source_name,s.points FROM stats s JOIN players p USING(player_id) WHERE s.dataset_key=:d ORDER BY s.points DESC,p.source_name LIMIT 3'),{'d':DATASET_KEY}).all()
        comparison=conn.execute(text('SELECT t.team_code, count(*) AS players, round(avg(s.points),2) AS avg_points FROM stats s JOIN teams t USING(team_code) WHERE s.dataset_key=:d GROUP BY t.team_code ORDER BY avg_points DESC,t.team_code LIMIT 3'),{'d':DATASET_KEY}).all()
        totals=conn.execute(text('SELECT count(*) AS n, round(avg(points),2) AS avg_points, max(points) AS max_points FROM stats WHERE dataset_key=:d'),{'d':DATASET_KEY}).mappings().one()
    if rejects:
        rej_path=Path(args.rejects or 'rejected_rows.csv')
        pd.DataFrame(rejects).to_csv(rej_path,index=False)
    else: rej_path=None
    print(json.dumps({'database_dialect':engine.dialect.name,'input_rows':len(frame),'valid_rows':len(valid),'rejected_rows':len(rejects),'reject_report':str(rej_path) if rej_path else None,'players_total':players_n,'teams_total':teams_n,'stats_for_dataset':stats_n,'matches_total':matches_n,'reports_upserted':reports_n,'test_queries':{'top_3_players_by_points':[{'player':r[0],'points':int(r[1])} for r in leaders],'top_3_teams_by_average_player_points':[{'team':r[0].strip(),'players':int(r[1]),'avg_points':float(r[2])} for r in comparison],'dataset_points_summary':{'players':int(totals['n']),'avg_points':float(totals['avg_points']),'max_points':int(totals['max_points'])}}},ensure_ascii=False,default=str,indent=2))
    engine.dispose()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xlsx',default=str(DEFAULT_XLSX),help='Classeur Excel source')
    ap.add_argument('--database-url',help='URL SQLAlchemy (sinon variable DATABASE_URL)')
    ap.add_argument('--skip-reports',action='store_true',help='Ne pas ingérer les quatre transcriptions Reddit')
    ap.add_argument('--rejects',help='Chemin du CSV des lignes rejetées')
    args=ap.parse_args()
    try: run(args)
    except Exception as exc:
        print(f'ERREUR: {exc}',file=sys.stderr); raise
if __name__=='__main__': main()
