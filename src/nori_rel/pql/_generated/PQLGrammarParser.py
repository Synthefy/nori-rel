# Derived from NVIDIA's Apache-2.0 PQL grammar; modified by Synthefy.
# See ../_vendor/kumo/LICENSE, NOTICE and ../grammar/PQLGrammar.g4.
# Generated from PQLGrammar.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,36,238,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,1,0,1,0,1,0,1,
        0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,54,8,1,1,2,1,2,
        1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,3,5,71,8,5,3,
        5,73,8,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,3,6,85,8,6,1,6,
        1,6,1,6,1,6,1,6,1,6,5,6,93,8,6,10,6,12,6,96,9,6,1,7,3,7,99,8,7,1,
        7,1,7,3,7,103,8,7,1,8,1,8,1,8,1,8,3,8,109,8,8,1,9,1,9,3,9,113,8,
        9,1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,11,1,11,1,12,1,12,1,
        12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,3,
        12,140,8,12,1,12,1,12,1,12,1,12,1,12,1,12,5,12,148,8,12,10,12,12,
        12,151,9,12,1,13,1,13,1,13,1,13,3,13,157,8,13,1,13,1,13,1,13,1,13,
        1,13,1,13,1,13,1,13,1,13,1,13,3,13,169,8,13,1,13,1,13,1,13,1,13,
        1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,3,13,183,8,13,1,13,1,13,
        3,13,187,8,13,1,14,1,14,1,14,1,14,1,15,1,15,1,16,1,16,3,16,197,8,
        16,1,16,1,16,1,16,3,16,202,8,16,1,16,1,16,1,16,1,16,3,16,208,8,16,
        1,17,1,17,1,17,1,17,4,17,214,8,17,11,17,12,17,215,1,17,1,17,1,17,
        1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,3,17,232,
        8,17,1,18,1,18,3,18,236,8,18,1,18,0,2,12,24,19,0,2,4,6,8,10,12,14,
        16,18,20,22,24,26,28,30,32,34,36,0,5,1,0,13,14,1,0,8,9,1,0,27,28,
        1,0,29,30,1,0,5,6,260,0,38,1,0,0,0,2,53,1,0,0,0,4,55,1,0,0,0,6,57,
        1,0,0,0,8,60,1,0,0,0,10,72,1,0,0,0,12,84,1,0,0,0,14,102,1,0,0,0,
        16,108,1,0,0,0,18,112,1,0,0,0,20,114,1,0,0,0,22,118,1,0,0,0,24,139,
        1,0,0,0,26,186,1,0,0,0,28,188,1,0,0,0,30,192,1,0,0,0,32,207,1,0,
        0,0,34,231,1,0,0,0,36,233,1,0,0,0,38,39,5,7,0,0,39,40,3,10,5,0,40,
        41,3,2,1,0,41,42,7,0,0,0,42,43,3,16,8,0,43,44,3,18,9,0,44,45,5,0,
        0,1,45,1,1,0,0,0,46,54,3,8,4,0,47,48,3,4,2,0,48,49,3,6,3,0,49,54,
        1,0,0,0,50,54,3,6,3,0,51,54,3,4,2,0,52,54,1,0,0,0,53,46,1,0,0,0,
        53,47,1,0,0,0,53,50,1,0,0,0,53,51,1,0,0,0,53,52,1,0,0,0,54,3,1,0,
        0,0,55,56,7,1,0,0,56,5,1,0,0,0,57,58,5,10,0,0,58,59,5,31,0,0,59,
        7,1,0,0,0,60,61,5,11,0,0,61,62,5,31,0,0,62,63,5,12,0,0,63,9,1,0,
        0,0,64,73,3,24,12,0,65,73,3,26,13,0,66,73,3,30,15,0,67,70,3,12,6,
        0,68,69,5,21,0,0,69,71,3,32,16,0,70,68,1,0,0,0,70,71,1,0,0,0,71,
        73,1,0,0,0,72,64,1,0,0,0,72,65,1,0,0,0,72,66,1,0,0,0,72,67,1,0,0,
        0,73,11,1,0,0,0,74,75,6,6,-1,0,75,76,7,2,0,0,76,85,3,12,6,7,77,78,
        5,1,0,0,78,79,3,12,6,0,79,80,5,2,0,0,80,85,1,0,0,0,81,85,3,26,13,
        0,82,85,3,32,16,0,83,85,3,30,15,0,84,74,1,0,0,0,84,77,1,0,0,0,84,
        81,1,0,0,0,84,82,1,0,0,0,84,83,1,0,0,0,85,94,1,0,0,0,86,87,10,2,
        0,0,87,88,7,3,0,0,88,93,3,12,6,3,89,90,10,1,0,0,90,91,7,2,0,0,91,
        93,3,12,6,2,92,86,1,0,0,0,92,89,1,0,0,0,93,96,1,0,0,0,94,92,1,0,
        0,0,94,95,1,0,0,0,95,13,1,0,0,0,96,94,1,0,0,0,97,99,5,27,0,0,98,
        97,1,0,0,0,98,99,1,0,0,0,99,100,1,0,0,0,100,103,5,31,0,0,101,103,
        5,23,0,0,102,98,1,0,0,0,102,101,1,0,0,0,103,15,1,0,0,0,104,109,3,
        30,15,0,105,109,3,28,14,0,106,109,3,20,10,0,107,109,3,22,11,0,108,
        104,1,0,0,0,108,105,1,0,0,0,108,106,1,0,0,0,108,107,1,0,0,0,109,
        17,1,0,0,0,110,111,5,16,0,0,111,113,3,24,12,0,112,110,1,0,0,0,112,
        113,1,0,0,0,113,19,1,0,0,0,114,115,3,30,15,0,115,116,5,21,0,0,116,
        117,3,32,16,0,117,21,1,0,0,0,118,119,3,30,15,0,119,120,5,21,0,0,
        120,121,3,32,16,0,121,122,5,15,0,0,122,123,3,24,12,0,123,23,1,0,
        0,0,124,125,6,12,-1,0,125,126,3,26,13,0,126,127,5,21,0,0,127,128,
        3,32,16,0,128,140,1,0,0,0,129,130,3,30,15,0,130,131,5,21,0,0,131,
        132,3,32,16,0,132,140,1,0,0,0,133,134,5,19,0,0,134,140,3,24,12,4,
        135,136,5,1,0,0,136,137,3,24,12,0,137,138,5,2,0,0,138,140,1,0,0,
        0,139,124,1,0,0,0,139,129,1,0,0,0,139,133,1,0,0,0,139,135,1,0,0,
        0,140,149,1,0,0,0,141,142,10,3,0,0,142,143,5,17,0,0,143,148,3,24,
        12,4,144,145,10,2,0,0,145,146,5,18,0,0,146,148,3,24,12,3,147,141,
        1,0,0,0,147,144,1,0,0,0,148,151,1,0,0,0,149,147,1,0,0,0,149,150,
        1,0,0,0,150,25,1,0,0,0,151,149,1,0,0,0,152,153,5,20,0,0,153,156,
        5,1,0,0,154,157,3,30,15,0,155,157,3,28,14,0,156,154,1,0,0,0,156,
        155,1,0,0,0,157,158,1,0,0,0,158,159,5,3,0,0,159,160,3,14,7,0,160,
        161,5,3,0,0,161,162,3,14,7,0,162,163,5,2,0,0,163,187,1,0,0,0,164,
        165,5,20,0,0,165,168,5,1,0,0,166,169,3,30,15,0,167,169,3,28,14,0,
        168,166,1,0,0,0,168,167,1,0,0,0,169,170,1,0,0,0,170,171,5,3,0,0,
        171,172,3,14,7,0,172,173,5,3,0,0,173,174,3,14,7,0,174,175,5,3,0,
        0,175,176,5,24,0,0,176,177,5,2,0,0,177,187,1,0,0,0,178,179,5,20,
        0,0,179,182,5,1,0,0,180,183,3,30,15,0,181,183,3,28,14,0,182,180,
        1,0,0,0,182,181,1,0,0,0,183,184,1,0,0,0,184,185,5,2,0,0,185,187,
        1,0,0,0,186,152,1,0,0,0,186,164,1,0,0,0,186,178,1,0,0,0,187,27,1,
        0,0,0,188,189,3,30,15,0,189,190,5,15,0,0,190,191,3,24,12,0,191,29,
        1,0,0,0,192,193,7,4,0,0,193,31,1,0,0,0,194,208,5,4,0,0,195,197,5,
        27,0,0,196,195,1,0,0,0,196,197,1,0,0,0,197,198,1,0,0,0,198,208,5,
        31,0,0,199,208,5,33,0,0,200,202,5,27,0,0,201,200,1,0,0,0,201,202,
        1,0,0,0,202,203,1,0,0,0,203,208,5,32,0,0,204,208,3,36,18,0,205,208,
        5,22,0,0,206,208,3,34,17,0,207,194,1,0,0,0,207,196,1,0,0,0,207,199,
        1,0,0,0,207,201,1,0,0,0,207,204,1,0,0,0,207,205,1,0,0,0,207,206,
        1,0,0,0,208,33,1,0,0,0,209,213,5,1,0,0,210,211,3,32,16,0,211,212,
        5,3,0,0,212,214,1,0,0,0,213,210,1,0,0,0,214,215,1,0,0,0,215,213,
        1,0,0,0,215,216,1,0,0,0,216,217,1,0,0,0,217,218,3,32,16,0,218,219,
        5,2,0,0,219,232,1,0,0,0,220,221,5,1,0,0,221,222,3,32,16,0,222,223,
        5,2,0,0,223,224,6,17,-1,0,224,232,1,0,0,0,225,226,5,1,0,0,226,227,
        3,32,16,0,227,228,5,3,0,0,228,229,5,2,0,0,229,230,6,17,-1,0,230,
        232,1,0,0,0,231,209,1,0,0,0,231,220,1,0,0,0,231,225,1,0,0,0,232,
        35,1,0,0,0,233,235,5,34,0,0,234,236,5,35,0,0,235,234,1,0,0,0,235,
        236,1,0,0,0,236,37,1,0,0,0,23,53,70,72,84,92,94,98,102,108,112,139,
        147,149,156,168,182,186,196,201,207,215,231,235
    ]

class PQLGrammarParser ( Parser ):

    grammarFileName = "PQLGrammar.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "','", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'-'", "'+'", "'*'", "'/'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "BOOL", "FQN_COLUMN", "WILDCARD_COLUMN", "PREDICT", 
                      "CLASSIFY", "RANK", "TOP", "FORECAST", "TIMEFRAMES", 
                      "FOR_EACH", "FOR", "WHERE", "ASSUMING", "AND", "OR", 
                      "NOT", "AGGR", "REL_OP", "NULL", "NEG_INFTY", "TIME_UNIT", 
                      "ID", "QUOTED_ID", "MINUS", "PLUS", "STAR", "SLASH", 
                      "INT", "DECIMAL", "STR", "DATE", "TIME", "WS" ]

    RULE_prog = 0
    RULE_problem_spec = 1
    RULE_problem_type = 2
    RULE_top_k = 3
    RULE_forecast = 4
    RULE_target = 5
    RULE_expression = 6
    RULE_offset = 7
    RULE_entity = 8
    RULE_whatif = 9
    RULE_entity_list = 10
    RULE_filtered_entity_list = 11
    RULE_condition = 12
    RULE_aggregation = 13
    RULE_filtered_column = 14
    RULE_column = 15
    RULE_constant = 16
    RULE_array = 17
    RULE_datetime = 18

    ruleNames =  [ "prog", "problem_spec", "problem_type", "top_k", "forecast", 
                   "target", "expression", "offset", "entity", "whatif", 
                   "entity_list", "filtered_entity_list", "condition", "aggregation", 
                   "filtered_column", "column", "constant", "array", "datetime" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    BOOL=4
    FQN_COLUMN=5
    WILDCARD_COLUMN=6
    PREDICT=7
    CLASSIFY=8
    RANK=9
    TOP=10
    FORECAST=11
    TIMEFRAMES=12
    FOR_EACH=13
    FOR=14
    WHERE=15
    ASSUMING=16
    AND=17
    OR=18
    NOT=19
    AGGR=20
    REL_OP=21
    NULL=22
    NEG_INFTY=23
    TIME_UNIT=24
    ID=25
    QUOTED_ID=26
    MINUS=27
    PLUS=28
    STAR=29
    SLASH=30
    INT=31
    DECIMAL=32
    STR=33
    DATE=34
    TIME=35
    WS=36

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PREDICT(self):
            return self.getToken(PQLGrammarParser.PREDICT, 0)

        def target(self):
            return self.getTypedRuleContext(PQLGrammarParser.TargetContext,0)


        def problem_spec(self):
            return self.getTypedRuleContext(PQLGrammarParser.Problem_specContext,0)


        def entity(self):
            return self.getTypedRuleContext(PQLGrammarParser.EntityContext,0)


        def whatif(self):
            return self.getTypedRuleContext(PQLGrammarParser.WhatifContext,0)


        def EOF(self):
            return self.getToken(PQLGrammarParser.EOF, 0)

        def FOR_EACH(self):
            return self.getToken(PQLGrammarParser.FOR_EACH, 0)

        def FOR(self):
            return self.getToken(PQLGrammarParser.FOR, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_prog




    def prog(self):

        localctx = PQLGrammarParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            self.match(PQLGrammarParser.PREDICT)
            self.state = 39
            self.target()
            self.state = 40
            self.problem_spec()
            self.state = 41
            _la = self._input.LA(1)
            if not(_la==13 or _la==14):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 42
            self.entity()
            self.state = 43
            self.whatif()
            self.state = 44
            self.match(PQLGrammarParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Problem_specContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def forecast(self):
            return self.getTypedRuleContext(PQLGrammarParser.ForecastContext,0)


        def problem_type(self):
            return self.getTypedRuleContext(PQLGrammarParser.Problem_typeContext,0)


        def top_k(self):
            return self.getTypedRuleContext(PQLGrammarParser.Top_kContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_problem_spec




    def problem_spec(self):

        localctx = PQLGrammarParser.Problem_specContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_problem_spec)
        try:
            self.state = 53
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 46
                self.forecast()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 47
                self.problem_type()
                self.state = 48
                self.top_k()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 50
                self.top_k()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 51
                self.problem_type()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Problem_typeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def RANK(self):
            return self.getToken(PQLGrammarParser.RANK, 0)

        def CLASSIFY(self):
            return self.getToken(PQLGrammarParser.CLASSIFY, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_problem_type




    def problem_type(self):

        localctx = PQLGrammarParser.Problem_typeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_problem_type)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 55
            _la = self._input.LA(1)
            if not(_la==8 or _la==9):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Top_kContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TOP(self):
            return self.getToken(PQLGrammarParser.TOP, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_top_k




    def top_k(self):

        localctx = PQLGrammarParser.Top_kContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_top_k)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 57
            self.match(PQLGrammarParser.TOP)
            self.state = 58
            self.match(PQLGrammarParser.INT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ForecastContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FORECAST(self):
            return self.getToken(PQLGrammarParser.FORECAST, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def TIMEFRAMES(self):
            return self.getToken(PQLGrammarParser.TIMEFRAMES, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_forecast




    def forecast(self):

        localctx = PQLGrammarParser.ForecastContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_forecast)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 60
            self.match(PQLGrammarParser.FORECAST)
            self.state = 61
            self.match(PQLGrammarParser.INT)
            self.state = 62
            self.match(PQLGrammarParser.TIMEFRAMES)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TargetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def aggregation(self):
            return self.getTypedRuleContext(PQLGrammarParser.AggregationContext,0)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def expression(self):
            return self.getTypedRuleContext(PQLGrammarParser.ExpressionContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_target




    def target(self):

        localctx = PQLGrammarParser.TargetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_target)
        self._la = 0 # Token type
        try:
            self.state = 72
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 64
                self.condition(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 65
                self.aggregation()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 66
                self.column()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 67
                self.expression(0)
                self.state = 70
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==21:
                    self.state = 68
                    self.match(PQLGrammarParser.REL_OP)
                    self.state = 69
                    self.constant()


                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expression(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.ExpressionContext,i)


        def MINUS(self):
            return self.getToken(PQLGrammarParser.MINUS, 0)

        def PLUS(self):
            return self.getToken(PQLGrammarParser.PLUS, 0)

        def aggregation(self):
            return self.getTypedRuleContext(PQLGrammarParser.AggregationContext,0)


        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def STAR(self):
            return self.getToken(PQLGrammarParser.STAR, 0)

        def SLASH(self):
            return self.getToken(PQLGrammarParser.SLASH, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_expression



    def expression(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PQLGrammarParser.ExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 12
        self.enterRecursionRule(localctx, 12, self.RULE_expression, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 84
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.state = 75
                _la = self._input.LA(1)
                if not(_la==27 or _la==28):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 76
                self.expression(7)
                pass

            elif la_ == 2:
                self.state = 77
                self.match(PQLGrammarParser.T__0)
                self.state = 78
                self.expression(0)
                self.state = 79
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 3:
                self.state = 81
                self.aggregation()
                pass

            elif la_ == 4:
                self.state = 82
                self.constant()
                pass

            elif la_ == 5:
                self.state = 83
                self.column()
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 94
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,5,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 92
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
                    if la_ == 1:
                        localctx = PQLGrammarParser.ExpressionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 86
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 87
                        _la = self._input.LA(1)
                        if not(_la==29 or _la==30):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 88
                        self.expression(3)
                        pass

                    elif la_ == 2:
                        localctx = PQLGrammarParser.ExpressionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expression)
                        self.state = 89
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                        self.state = 90
                        _la = self._input.LA(1)
                        if not(_la==27 or _la==28):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 91
                        self.expression(2)
                        pass

             
                self.state = 96
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class OffsetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def MINUS(self):
            return self.getToken(PQLGrammarParser.MINUS, 0)

        def NEG_INFTY(self):
            return self.getToken(PQLGrammarParser.NEG_INFTY, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_offset




    def offset(self):

        localctx = PQLGrammarParser.OffsetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_offset)
        self._la = 0 # Token type
        try:
            self.state = 102
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [27, 31]:
                self.enterOuterAlt(localctx, 1)
                self.state = 98
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==27:
                    self.state = 97
                    self.match(PQLGrammarParser.MINUS)


                self.state = 100
                self.match(PQLGrammarParser.INT)
                pass
            elif token in [23]:
                self.enterOuterAlt(localctx, 2)
                self.state = 101
                self.match(PQLGrammarParser.NEG_INFTY)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EntityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def filtered_column(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_columnContext,0)


        def entity_list(self):
            return self.getTypedRuleContext(PQLGrammarParser.Entity_listContext,0)


        def filtered_entity_list(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_entity_listContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_entity




    def entity(self):

        localctx = PQLGrammarParser.EntityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_entity)
        try:
            self.state = 108
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 104
                self.column()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 105
                self.filtered_column()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 106
                self.entity_list()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 107
                self.filtered_entity_list()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class WhatifContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ASSUMING(self):
            return self.getToken(PQLGrammarParser.ASSUMING, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_whatif




    def whatif(self):

        localctx = PQLGrammarParser.WhatifContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_whatif)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 112
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==16:
                self.state = 110
                self.match(PQLGrammarParser.ASSUMING)
                self.state = 111
                self.condition(0)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Entity_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_entity_list




    def entity_list(self):

        localctx = PQLGrammarParser.Entity_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 114
            self.column()
            self.state = 115
            self.match(PQLGrammarParser.REL_OP)
            self.state = 116
            self.constant()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Filtered_entity_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def WHERE(self):
            return self.getToken(PQLGrammarParser.WHERE, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_filtered_entity_list




    def filtered_entity_list(self):

        localctx = PQLGrammarParser.Filtered_entity_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_filtered_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 118
            self.column()
            self.state = 119
            self.match(PQLGrammarParser.REL_OP)
            self.state = 120
            self.constant()
            self.state = 121
            self.match(PQLGrammarParser.WHERE)
            self.state = 122
            self.condition(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConditionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def aggregation(self):
            return self.getTypedRuleContext(PQLGrammarParser.AggregationContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def NOT(self):
            return self.getToken(PQLGrammarParser.NOT, 0)

        def condition(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.ConditionContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,i)


        def AND(self):
            return self.getToken(PQLGrammarParser.AND, 0)

        def OR(self):
            return self.getToken(PQLGrammarParser.OR, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_condition



    def condition(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PQLGrammarParser.ConditionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 24
        self.enterRecursionRule(localctx, 24, self.RULE_condition, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 139
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [20]:
                self.state = 125
                self.aggregation()
                self.state = 126
                self.match(PQLGrammarParser.REL_OP)
                self.state = 127
                self.constant()
                pass
            elif token in [5, 6]:
                self.state = 129
                self.column()
                self.state = 130
                self.match(PQLGrammarParser.REL_OP)
                self.state = 131
                self.constant()
                pass
            elif token in [19]:
                self.state = 133
                self.match(PQLGrammarParser.NOT)
                self.state = 134
                self.condition(4)
                pass
            elif token in [1]:
                self.state = 135
                self.match(PQLGrammarParser.T__0)
                self.state = 136
                self.condition(0)
                self.state = 137
                self.match(PQLGrammarParser.T__1)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 149
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,12,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 147
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,11,self._ctx)
                    if la_ == 1:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 141
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 142
                        self.match(PQLGrammarParser.AND)
                        self.state = 143
                        self.condition(4)
                        pass

                    elif la_ == 2:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 144
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 145
                        self.match(PQLGrammarParser.OR)
                        self.state = 146
                        self.condition(3)
                        pass

             
                self.state = 151
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class AggregationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def AGGR(self):
            return self.getToken(PQLGrammarParser.AGGR, 0)

        def offset(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.OffsetContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.OffsetContext,i)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def filtered_column(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_columnContext,0)


        def TIME_UNIT(self):
            return self.getToken(PQLGrammarParser.TIME_UNIT, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_aggregation




    def aggregation(self):

        localctx = PQLGrammarParser.AggregationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_aggregation)
        try:
            self.state = 186
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,16,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 152
                self.match(PQLGrammarParser.AGGR)
                self.state = 153
                self.match(PQLGrammarParser.T__0)
                self.state = 156
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
                if la_ == 1:
                    self.state = 154
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 155
                    self.filtered_column()
                    pass


                self.state = 158
                self.match(PQLGrammarParser.T__2)
                self.state = 159
                self.offset()
                self.state = 160
                self.match(PQLGrammarParser.T__2)
                self.state = 161
                self.offset()
                self.state = 162
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 164
                self.match(PQLGrammarParser.AGGR)
                self.state = 165
                self.match(PQLGrammarParser.T__0)
                self.state = 168
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
                if la_ == 1:
                    self.state = 166
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 167
                    self.filtered_column()
                    pass


                self.state = 170
                self.match(PQLGrammarParser.T__2)
                self.state = 171
                self.offset()
                self.state = 172
                self.match(PQLGrammarParser.T__2)
                self.state = 173
                self.offset()
                self.state = 174
                self.match(PQLGrammarParser.T__2)
                self.state = 175
                self.match(PQLGrammarParser.TIME_UNIT)
                self.state = 176
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 178
                self.match(PQLGrammarParser.AGGR)
                self.state = 179
                self.match(PQLGrammarParser.T__0)
                self.state = 182
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
                if la_ == 1:
                    self.state = 180
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 181
                    self.filtered_column()
                    pass


                self.state = 184
                self.match(PQLGrammarParser.T__1)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Filtered_columnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def WHERE(self):
            return self.getToken(PQLGrammarParser.WHERE, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_filtered_column




    def filtered_column(self):

        localctx = PQLGrammarParser.Filtered_columnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_filtered_column)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 188
            self.column()
            self.state = 189
            self.match(PQLGrammarParser.WHERE)
            self.state = 190
            self.condition(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ColumnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FQN_COLUMN(self):
            return self.getToken(PQLGrammarParser.FQN_COLUMN, 0)

        def WILDCARD_COLUMN(self):
            return self.getToken(PQLGrammarParser.WILDCARD_COLUMN, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_column




    def column(self):

        localctx = PQLGrammarParser.ColumnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_column)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 192
            _la = self._input.LA(1)
            if not(_la==5 or _la==6):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstantContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BOOL(self):
            return self.getToken(PQLGrammarParser.BOOL, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def MINUS(self):
            return self.getToken(PQLGrammarParser.MINUS, 0)

        def STR(self):
            return self.getToken(PQLGrammarParser.STR, 0)

        def DECIMAL(self):
            return self.getToken(PQLGrammarParser.DECIMAL, 0)

        def datetime(self):
            return self.getTypedRuleContext(PQLGrammarParser.DatetimeContext,0)


        def NULL(self):
            return self.getToken(PQLGrammarParser.NULL, 0)

        def array(self):
            return self.getTypedRuleContext(PQLGrammarParser.ArrayContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_constant




    def constant(self):

        localctx = PQLGrammarParser.ConstantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_constant)
        self._la = 0 # Token type
        try:
            self.state = 207
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 194
                self.match(PQLGrammarParser.BOOL)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 196
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==27:
                    self.state = 195
                    self.match(PQLGrammarParser.MINUS)


                self.state = 198
                self.match(PQLGrammarParser.INT)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 199
                self.match(PQLGrammarParser.STR)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 201
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==27:
                    self.state = 200
                    self.match(PQLGrammarParser.MINUS)


                self.state = 203
                self.match(PQLGrammarParser.DECIMAL)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 204
                self.datetime()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 205
                self.match(PQLGrammarParser.NULL)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 206
                self.array()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArrayContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constant(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.ConstantContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,i)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_array




    def array(self):

        localctx = PQLGrammarParser.ArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_array)
        try:
            self.state = 231
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,21,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 209
                self.match(PQLGrammarParser.T__0)
                self.state = 213 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 210
                        self.constant()
                        self.state = 211
                        self.match(PQLGrammarParser.T__2)

                    else:
                        raise NoViableAltException(self)
                    self.state = 215 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,20,self._ctx)

                self.state = 217
                self.constant()
                self.state = 218
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 220
                self.match(PQLGrammarParser.T__0)
                self.state = 221
                self.constant()
                self.state = 222
                self.match(PQLGrammarParser.T__1)
                raise RecognitionException("Array with single element is not supported")
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 225
                self.match(PQLGrammarParser.T__0)
                self.state = 226
                self.constant()
                self.state = 227
                self.match(PQLGrammarParser.T__2)
                self.state = 228
                self.match(PQLGrammarParser.T__1)
                raise RecognitionException("Array with single element is not supported")
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DatetimeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DATE(self):
            return self.getToken(PQLGrammarParser.DATE, 0)

        def TIME(self):
            return self.getToken(PQLGrammarParser.TIME, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_datetime




    def datetime(self):

        localctx = PQLGrammarParser.DatetimeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_datetime)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 233
            self.match(PQLGrammarParser.DATE)
            self.state = 235
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,22,self._ctx)
            if la_ == 1:
                self.state = 234
                self.match(PQLGrammarParser.TIME)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[6] = self.expression_sempred
        self._predicates[12] = self.condition_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expression_sempred(self, localctx:ExpressionContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 2)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 1)
         

    def condition_sempred(self, localctx:ConditionContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 3)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 2)
         




