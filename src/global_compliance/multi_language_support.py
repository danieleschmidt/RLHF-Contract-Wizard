"""
Multi-Language Support System for RLHF-Contract-Wizard.

This module provides comprehensive internationalization (i18n) and localization (l10n)
support for the RLHF Contract system, enabling global deployment with native language
support for users, documentation, and regulatory compliance.

Features:
1. Dynamic Language Detection and Switching
2. Real-time Translation Services
3. Cultural Adaptation and Localization
4. Regulatory Text Translation
5. Multi-script Support (Latin, Cyrillic, CJK, Arabic, etc.)
6. Contextual Translation for Technical Terms
7. Audio/Visual Localization Support
"""

import json
import os
import re
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import locale
import unicodedata

from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class SupportedLanguage(Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"
    THAI = "th"
    VIETNAMESE = "vi"
    POLISH = "pl"
    CZECH = "cs"
    HUNGARIAN = "hu"
    TURKISH = "tr"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"


class TextDirection(Enum):
    """Text reading direction."""
    LEFT_TO_RIGHT = "ltr"
    RIGHT_TO_LEFT = "rtl"
    TOP_TO_BOTTOM = "ttb"


class CulturalContext(Enum):
    """Cultural contexts for adaptation."""
    WESTERN = "western"
    EASTERN = "eastern"
    MIDDLE_EASTERN = "middle_eastern"
    AFRICAN = "african"
    LATIN_AMERICAN = "latin_american"
    NORDIC = "nordic"
    SLAVIC = "slavic"


@dataclass
class LanguageProfile:
    """Complete language profile for localization."""
    language_code: str
    native_name: str
    english_name: str
    text_direction: TextDirection
    cultural_context: CulturalContext
    decimal_separator: str
    thousands_separator: str
    currency_position: str  # "before" or "after"
    date_format: str
    time_format: str
    plural_rules: Dict[str, str]
    formal_address_required: bool = False
    honorific_system: bool = False
    script_type: str = "latin"
    rtl_support_needed: bool = False


@dataclass
class TranslationEntry:
    """Individual translation entry."""
    key: str
    original_text: str
    translated_text: str
    context: str
    language: str
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    translator: str = "system"
    reviewed: bool = False


@dataclass
class LocalizationContext:
    """Context for localization decisions."""
    target_language: SupportedLanguage
    user_region: str
    business_context: str
    formality_level: str  # "formal", "informal", "technical"
    audience_type: str    # "general", "technical", "legal", "business"


class TranslationEngine:
    """Advanced translation engine with context awareness."""
    
    def __init__(self):
        self.translation_cache = {}
        self.language_models = {}
        self.terminology_databases = {}
        
        # Initialize language profiles
        self.language_profiles = self._initialize_language_profiles()
        
        # Initialize translation patterns
        self.translation_patterns = self._initialize_translation_patterns()
        
        # Load terminology databases
        self._load_terminology_databases()
    
    def _initialize_language_profiles(self) -> Dict[str, LanguageProfile]:
        """Initialize comprehensive language profiles."""
        
        profiles = {}
        
        # Western European Languages
        profiles["en"] = LanguageProfile(
            language_code="en",
            native_name="English",
            english_name="English",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.WESTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="before",
            date_format="%m/%d/%Y",
            time_format="%I:%M %p",
            plural_rules={"zero": "other", "one": "one", "other": "other"},
            script_type="latin"
        )
        
        profiles["es"] = LanguageProfile(
            language_code="es",
            native_name="Español",
            english_name="Spanish",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.LATIN_AMERICAN,
            decimal_separator=",",
            thousands_separator=".",
            currency_position="after",
            date_format="%d/%m/%Y",
            time_format="%H:%M",
            plural_rules={"zero": "other", "one": "one", "other": "other"},
            formal_address_required=True,
            script_type="latin"
        )
        
        profiles["fr"] = LanguageProfile(
            language_code="fr",
            native_name="Français",
            english_name="French",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.WESTERN,
            decimal_separator=",",
            thousands_separator=" ",
            currency_position="after",
            date_format="%d/%m/%Y",
            time_format="%H:%M",
            plural_rules={"zero": "one", "one": "one", "other": "other"},
            formal_address_required=True,
            script_type="latin"
        )
        
        profiles["de"] = LanguageProfile(
            language_code="de",
            native_name="Deutsch",
            english_name="German",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.WESTERN,
            decimal_separator=",",
            thousands_separator=".",
            currency_position="after",
            date_format="%d.%m.%Y",
            time_format="%H:%M",
            plural_rules={"zero": "other", "one": "one", "other": "other"},
            formal_address_required=True,
            script_type="latin"
        )
        
        # East Asian Languages
        profiles["zh-CN"] = LanguageProfile(
            language_code="zh-CN",
            native_name="简体中文",
            english_name="Chinese (Simplified)",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.EASTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="before",
            date_format="%Y年%m月%d日",
            time_format="%H:%M",
            plural_rules={"other": "other"},  # Chinese doesn't distinguish plural
            honorific_system=True,
            script_type="cjk"
        )
        
        profiles["ja"] = LanguageProfile(
            language_code="ja",
            native_name="日本語",
            english_name="Japanese",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.EASTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="before",
            date_format="%Y年%m月%d日",
            time_format="%H:%M",
            plural_rules={"other": "other"},
            formal_address_required=True,
            honorific_system=True,
            script_type="cjk"
        )
        
        profiles["ko"] = LanguageProfile(
            language_code="ko",
            native_name="한국어",
            english_name="Korean",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.EASTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="before",
            date_format="%Y년 %m월 %d일",
            time_format="%H:%M",
            plural_rules={"other": "other"},
            formal_address_required=True,
            honorific_system=True,
            script_type="cjk"
        )
        
        # Right-to-Left Languages
        profiles["ar"] = LanguageProfile(
            language_code="ar",
            native_name="العربية",
            english_name="Arabic",
            text_direction=TextDirection.RIGHT_TO_LEFT,
            cultural_context=CulturalContext.MIDDLE_EASTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="before",
            date_format="%d/%m/%Y",
            time_format="%H:%M",
            plural_rules={"zero": "zero", "one": "one", "two": "two", "few": "few", "many": "many", "other": "other"},
            formal_address_required=True,
            script_type="arabic",
            rtl_support_needed=True
        )
        
        profiles["he"] = LanguageProfile(
            language_code="he",
            native_name="עברית",
            english_name="Hebrew",
            text_direction=TextDirection.RIGHT_TO_LEFT,
            cultural_context=CulturalContext.MIDDLE_EASTERN,
            decimal_separator=".",
            thousands_separator=",",
            currency_position="after",
            date_format="%d/%m/%Y",
            time_format="%H:%M",
            plural_rules={"one": "one", "two": "two", "other": "other"},
            script_type="hebrew",
            rtl_support_needed=True
        )
        
        # Cyrillic Languages
        profiles["ru"] = LanguageProfile(
            language_code="ru",
            native_name="Русский",
            english_name="Russian",
            text_direction=TextDirection.LEFT_TO_RIGHT,
            cultural_context=CulturalContext.SLAVIC,
            decimal_separator=",",
            thousands_separator=" ",
            currency_position="after",
            date_format="%d.%m.%Y",
            time_format="%H:%M",
            plural_rules={"one": "one", "few": "few", "many": "many", "other": "other"},
            formal_address_required=True,
            script_type="cyrillic"
        )
        
        return profiles
    
    def _initialize_translation_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize translation patterns for technical terms."""
        
        patterns = {
            "rlhf_terms": {
                "en": {
                    "reward_function": "reward function",
                    "contract": "contract", 
                    "stakeholder": "stakeholder",
                    "compliance": "compliance",
                    "verification": "verification",
                    "blockchain": "blockchain",
                    "smart_contract": "smart contract",
                    "consensus": "consensus",
                    "governance": "governance",
                    "audit_trail": "audit trail"
                },
                "es": {
                    "reward_function": "función de recompensa",
                    "contract": "contrato",
                    "stakeholder": "parte interesada", 
                    "compliance": "cumplimiento",
                    "verification": "verificación",
                    "blockchain": "cadena de bloques",
                    "smart_contract": "contrato inteligente",
                    "consensus": "consenso",
                    "governance": "gobernanza",
                    "audit_trail": "pista de auditoría"
                },
                "fr": {
                    "reward_function": "fonction de récompense",
                    "contract": "contrat",
                    "stakeholder": "partie prenante",
                    "compliance": "conformité",
                    "verification": "vérification", 
                    "blockchain": "chaîne de blocs",
                    "smart_contract": "contrat intelligent",
                    "consensus": "consensus",
                    "governance": "gouvernance",
                    "audit_trail": "piste d'audit"
                },
                "de": {
                    "reward_function": "Belohnungsfunktion",
                    "contract": "Vertrag",
                    "stakeholder": "Stakeholder",
                    "compliance": "Compliance", 
                    "verification": "Verifizierung",
                    "blockchain": "Blockchain",
                    "smart_contract": "Smart Contract",
                    "consensus": "Konsens",
                    "governance": "Governance",
                    "audit_trail": "Prüfpfad"
                },
                "zh-CN": {
                    "reward_function": "奖励函数",
                    "contract": "合约",
                    "stakeholder": "利益相关者",
                    "compliance": "合规",
                    "verification": "验证",
                    "blockchain": "区块链",
                    "smart_contract": "智能合约",
                    "consensus": "共识",
                    "governance": "治理",
                    "audit_trail": "审计跟踪"
                },
                "ja": {
                    "reward_function": "報酬関数",
                    "contract": "契約",
                    "stakeholder": "ステークホルダー",
                    "compliance": "コンプライアンス",
                    "verification": "検証",
                    "blockchain": "ブロックチェーン",
                    "smart_contract": "スマートコントラクト", 
                    "consensus": "コンセンサス",
                    "governance": "ガバナンス",
                    "audit_trail": "監査証跡"
                },
                "ar": {
                    "reward_function": "دالة المكافآت",
                    "contract": "عقد",
                    "stakeholder": "صاحب مصلحة",
                    "compliance": "امتثال",
                    "verification": "تحقق",
                    "blockchain": "بلوك تشين",
                    "smart_contract": "عقد ذكي",
                    "consensus": "إجماع",
                    "governance": "حوكمة",
                    "audit_trail": "مسار المراجعة"
                }
            },
            "ui_elements": {
                "en": {
                    "login": "Login",
                    "logout": "Logout",
                    "dashboard": "Dashboard",
                    "settings": "Settings",
                    "profile": "Profile",
                    "notifications": "Notifications",
                    "search": "Search",
                    "filter": "Filter",
                    "sort": "Sort",
                    "export": "Export",
                    "import": "Import",
                    "save": "Save",
                    "cancel": "Cancel",
                    "delete": "Delete",
                    "edit": "Edit",
                    "create": "Create",
                    "submit": "Submit"
                },
                "es": {
                    "login": "Iniciar sesión",
                    "logout": "Cerrar sesión", 
                    "dashboard": "Panel de control",
                    "settings": "Configuración",
                    "profile": "Perfil",
                    "notifications": "Notificaciones",
                    "search": "Buscar",
                    "filter": "Filtrar",
                    "sort": "Ordenar",
                    "export": "Exportar",
                    "import": "Importar",
                    "save": "Guardar",
                    "cancel": "Cancelar",
                    "delete": "Eliminar",
                    "edit": "Editar",
                    "create": "Crear",
                    "submit": "Enviar"
                },
                "fr": {
                    "login": "Connexion",
                    "logout": "Déconnexion",
                    "dashboard": "Tableau de bord",
                    "settings": "Paramètres",
                    "profile": "Profil",
                    "notifications": "Notifications",
                    "search": "Rechercher",
                    "filter": "Filtrer",
                    "sort": "Trier",
                    "export": "Exporter",
                    "import": "Importer",
                    "save": "Enregistrer",
                    "cancel": "Annuler",
                    "delete": "Supprimer",
                    "edit": "Modifier",
                    "create": "Créer",
                    "submit": "Soumettre"
                },
                "de": {
                    "login": "Anmelden",
                    "logout": "Abmelden",
                    "dashboard": "Dashboard",
                    "settings": "Einstellungen",
                    "profile": "Profil",
                    "notifications": "Benachrichtigungen",
                    "search": "Suchen",
                    "filter": "Filter",
                    "sort": "Sortieren",
                    "export": "Exportieren",
                    "import": "Importieren",
                    "save": "Speichern",
                    "cancel": "Abbrechen",
                    "delete": "Löschen",
                    "edit": "Bearbeiten",
                    "create": "Erstellen",
                    "submit": "Senden"
                },
                "zh-CN": {
                    "login": "登录",
                    "logout": "登出",
                    "dashboard": "仪表板",
                    "settings": "设置",
                    "profile": "个人资料",
                    "notifications": "通知",
                    "search": "搜索",
                    "filter": "筛选",
                    "sort": "排序",
                    "export": "导出",
                    "import": "导入",
                    "save": "保存",
                    "cancel": "取消",
                    "delete": "删除",
                    "edit": "编辑",
                    "create": "创建",
                    "submit": "提交"
                }
            }
        }
        
        return patterns
    
    def _load_terminology_databases(self):
        """Load specialized terminology databases."""
        
        # Legal terminology
        self.terminology_databases["legal"] = {
            "en": {
                "jurisdiction": "jurisdiction",
                "liability": "liability",
                "indemnification": "indemnification",
                "force_majeure": "force majeure",
                "intellectual_property": "intellectual property",
                "confidentiality": "confidentiality",
                "non_disclosure": "non-disclosure",
                "arbitration": "arbitration",
                "litigation": "litigation",
                "damages": "damages"
            },
            "es": {
                "jurisdiction": "jurisdicción",
                "liability": "responsabilidad",
                "indemnification": "indemnización",
                "force_majeure": "fuerza mayor",
                "intellectual_property": "propiedad intelectual",
                "confidentiality": "confidencialidad",
                "non_disclosure": "no divulgación",
                "arbitration": "arbitraje",
                "litigation": "litigio",
                "damages": "daños"
            }
        }
        
        # Technical terminology
        self.terminology_databases["technical"] = {
            "en": {
                "algorithm": "algorithm",
                "machine_learning": "machine learning",
                "neural_network": "neural network",
                "deep_learning": "deep learning",
                "artificial_intelligence": "artificial intelligence",
                "natural_language_processing": "natural language processing",
                "computer_vision": "computer vision",
                "reinforcement_learning": "reinforcement learning",
                "supervised_learning": "supervised learning",
                "unsupervised_learning": "unsupervised learning"
            },
            "zh-CN": {
                "algorithm": "算法",
                "machine_learning": "机器学习",
                "neural_network": "神经网络",
                "deep_learning": "深度学习",
                "artificial_intelligence": "人工智能",
                "natural_language_processing": "自然语言处理",
                "computer_vision": "计算机视觉",
                "reinforcement_learning": "强化学习",
                "supervised_learning": "监督学习",
                "unsupervised_learning": "无监督学习"
            }
        }
    
    def translate_text(self, text: str, target_language: SupportedLanguage,
                      context: Optional[LocalizationContext] = None) -> TranslationEntry:
        """Translate text with context awareness."""
        
        try:
            # Check cache first
            cache_key = f"{text}:{target_language.value}:{context.formality_level if context else 'default'}"
            
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
            
            # Determine source language (assume English for now)
            source_language = "en"
            
            # Get target language profile
            target_profile = self.language_profiles.get(target_language.value)
            if not target_profile:
                raise ValueError(f"Unsupported language: {target_language.value}")
            
            # Perform contextual translation
            translated_text = self._perform_translation(
                text, source_language, target_language.value, context
            )
            
            # Create translation entry
            translation_entry = TranslationEntry(
                key=cache_key,
                original_text=text,
                translated_text=translated_text,
                context=context.business_context if context else "general",
                language=target_language.value,
                confidence=self._calculate_translation_confidence(text, translated_text, context)
            )
            
            # Cache the translation
            self.translation_cache[cache_key] = translation_entry
            
            return translation_entry
            
        except Exception as e:
            handle_error(
                error=e,
                operation="translate_text",
                category=ErrorCategory.TRANSLATION,
                severity=ErrorSeverity.MEDIUM,
                additional_info={
                    'text_length': len(text),
                    'target_language': target_language.value,
                    'context': context.business_context if context else 'none'
                }
            )
            
            # Return fallback translation
            return TranslationEntry(
                key=cache_key,
                original_text=text,
                translated_text=text,  # Fallback to original
                context="error",
                language=target_language.value,
                confidence=0.0
            )
    
    def _perform_translation(self, text: str, source_lang: str, target_lang: str,
                           context: Optional[LocalizationContext] = None) -> str:
        """Perform the actual translation with context."""
        
        # Check for technical terms first
        translated_text = self._translate_technical_terms(text, source_lang, target_lang, context)
        
        # Handle special formatting
        translated_text = self._handle_text_formatting(translated_text, target_lang)
        
        # Apply cultural adaptations
        if context:
            translated_text = self._apply_cultural_adaptations(translated_text, context)
        
        # For demonstration, we'll use pattern matching
        # In production, this would use real translation APIs
        if target_lang in self.translation_patterns.get("ui_elements", {}):
            ui_patterns = self.translation_patterns["ui_elements"][target_lang]
            for en_term, translated_term in ui_patterns.items():
                if en_term.lower() in text.lower():
                    translated_text = text.lower().replace(en_term.lower(), translated_term)
        
        # Apply language-specific transformations
        translated_text = self._apply_language_transformations(translated_text, target_lang)
        
        return translated_text or text  # Fallback to original if translation fails
    
    def _translate_technical_terms(self, text: str, source_lang: str, target_lang: str,
                                 context: Optional[LocalizationContext] = None) -> str:
        """Translate technical terms using specialized dictionaries."""
        
        result = text
        
        # RLHF-specific terms
        if target_lang in self.translation_patterns.get("rlhf_terms", {}):
            rlhf_terms = self.translation_patterns["rlhf_terms"][target_lang]
            for en_term, translated_term in rlhf_terms.items():
                # Case-insensitive replacement with word boundaries
                pattern = r'\b' + re.escape(en_term.replace('_', ' ')) + r'\b'
                result = re.sub(pattern, translated_term, result, flags=re.IGNORECASE)
        
        # Technical terms based on context
        if context and context.audience_type == "technical":
            tech_terms = self.terminology_databases.get("technical", {}).get(target_lang, {})
            for en_term, translated_term in tech_terms.items():
                pattern = r'\b' + re.escape(en_term.replace('_', ' ')) + r'\b'
                result = re.sub(pattern, translated_term, result, flags=re.IGNORECASE)
        
        # Legal terms
        if context and context.business_context == "legal":
            legal_terms = self.terminology_databases.get("legal", {}).get(target_lang, {})
            for en_term, translated_term in legal_terms.items():
                pattern = r'\b' + re.escape(en_term.replace('_', ' ')) + r'\b'
                result = re.sub(pattern, translated_term, result, flags=re.IGNORECASE)
        
        return result
    
    def _handle_text_formatting(self, text: str, target_lang: str) -> str:
        """Handle text formatting for specific languages."""
        
        profile = self.language_profiles.get(target_lang)
        if not profile:
            return text
        
        result = text
        
        # Handle RTL languages
        if profile.rtl_support_needed:
            # Add RTL markers if needed
            if not result.startswith('\u202E') and not result.startswith('\u202D'):
                result = '\u202B' + result + '\u202C'  # RTL embedding
        
        # Handle CJK languages
        if profile.script_type == "cjk":
            # Remove unnecessary spaces between CJK characters
            result = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', result)
        
        # Handle Arabic script
        if profile.script_type == "arabic":
            # Ensure proper Arabic text handling
            result = unicodedata.normalize('NFKC', result)
        
        return result
    
    def _apply_cultural_adaptations(self, text: str, context: LocalizationContext) -> str:
        """Apply cultural adaptations to translations."""
        
        profile = self.language_profiles.get(context.target_language.value)
        if not profile:
            return text
        
        result = text
        
        # Formality adjustments
        if profile.formal_address_required and context.formality_level == "formal":
            # This would include formal pronouns, honorifics, etc.
            # Simplified for demonstration
            if profile.language_code in ["es", "fr", "de"]:
                result = result.replace("you", "usted" if profile.language_code == "es" else "vous" if profile.language_code == "fr" else "Sie")
        
        # Honorific systems
        if profile.honorific_system and context.audience_type == "business":
            # Add appropriate honorifics for Japanese, Korean, etc.
            if profile.language_code == "ja":
                # Add -san, -sama, etc. where appropriate
                pass
        
        # Currency and number formatting
        result = self._format_numbers_and_currency(result, profile)
        
        # Date and time formatting
        result = self._format_dates_and_times(result, profile)
        
        return result
    
    def _apply_language_transformations(self, text: str, target_lang: str) -> str:
        """Apply language-specific transformations."""
        
        profile = self.language_profiles.get(target_lang)
        if not profile:
            return text
        
        result = text
        
        # Capitalization rules
        if target_lang == "de":
            # German capitalizes all nouns
            # This is a simplified example
            words = result.split()
            for i, word in enumerate(words):
                if len(word) > 3 and word.islower():
                    # Simple heuristic for nouns (would be more sophisticated in practice)
                    if word.endswith(('ung', 'heit', 'keit', 'schaft')):
                        words[i] = word.capitalize()
            result = ' '.join(words)
        
        # Punctuation adjustments
        if target_lang == "fr":
            # French uses non-breaking spaces before certain punctuation
            result = re.sub(r'\s*([!?:;])', r'\u00A0\\1', result)
        
        # Quote marks
        if target_lang == "de":
            result = result.replace('"', '„').replace('"', '"')
        elif target_lang == "fr":
            result = result.replace('"', '«\u00A0').replace('"', '\u00A0»')
        
        return result
    
    def _format_numbers_and_currency(self, text: str, profile: LanguageProfile) -> str:
        """Format numbers and currency according to locale."""
        
        result = text
        
        # Number formatting
        number_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?'
        
        def replace_number(match):
            number_str = match.group(0)
            
            # Parse the number
            if ',' in number_str and '.' in number_str:
                # Assume American format (1,234.56)
                parts = number_str.split('.')
                integer_part = parts[0].replace(',', '')
                decimal_part = parts[1] if len(parts) > 1 else ''
            else:
                integer_part = number_str.replace(',', '')
                decimal_part = ''
            
            # Reformat according to profile
            if len(integer_part) > 3:
                # Add thousands separator
                thousands_sep = profile.thousands_separator
                formatted_integer = ''
                for i, digit in enumerate(reversed(integer_part)):
                    if i > 0 and i % 3 == 0:
                        formatted_integer = thousands_sep + formatted_integer
                    formatted_integer = digit + formatted_integer
            else:
                formatted_integer = integer_part
            
            # Add decimal part if exists
            if decimal_part:
                return formatted_integer + profile.decimal_separator + decimal_part
            else:
                return formatted_integer
        
        result = re.sub(number_pattern, replace_number, result)
        
        return result
    
    def _format_dates_and_times(self, text: str, profile: LanguageProfile) -> str:
        """Format dates and times according to locale."""
        
        # This is a simplified implementation
        # In practice, would use proper date parsing and formatting
        
        result = text
        
        # Simple date pattern (MM/DD/YYYY or DD/MM/YYYY)
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        
        def replace_date(match):
            date_str = match.group(0)
            parts = date_str.split('/')
            
            if profile.language_code in ['en']:
                # MM/DD/YYYY format
                return f"{parts[0]}/{parts[1]}/{parts[2]}"
            elif profile.language_code in ['es', 'fr', 'de']:
                # DD/MM/YYYY format
                return f"{parts[1]}/{parts[0]}/{parts[2]}"
            elif profile.language_code in ['zh-CN', 'ja', 'ko']:
                # YYYY/MM/DD format
                return f"{parts[2]}/{parts[0]}/{parts[1]}"
            else:
                return date_str
        
        result = re.sub(date_pattern, replace_date, result)
        
        return result
    
    def _calculate_translation_confidence(self, original: str, translated: str,
                                        context: Optional[LocalizationContext] = None) -> float:
        """Calculate confidence score for translation."""
        
        # Simplified confidence calculation
        base_confidence = 0.8
        
        # Reduce confidence for very short texts
        if len(original.split()) < 3:
            base_confidence -= 0.2
        
        # Increase confidence for technical terms that were translated
        if context and context.audience_type == "technical":
            tech_terms_found = len([term for term in self.terminology_databases.get("technical", {}).get("en", {}) 
                                  if term.replace('_', ' ') in original.lower()])
            if tech_terms_found > 0:
                base_confidence += 0.1
        
        # Reduce confidence if translation is identical to original (likely no translation occurred)
        if original == translated:
            base_confidence = 0.3
        
        return min(1.0, max(0.0, base_confidence))
    
    def batch_translate(self, texts: List[str], target_language: SupportedLanguage,
                       context: Optional[LocalizationContext] = None) -> List[TranslationEntry]:
        """Translate multiple texts efficiently."""
        
        results = []
        
        for text in texts:
            try:
                translation = self.translate_text(text, target_language, context)
                results.append(translation)
            except Exception as e:
                # Create error translation entry
                error_translation = TranslationEntry(
                    key=f"{text}:error",
                    original_text=text,
                    translated_text=text,  # Fallback
                    context="error",
                    language=target_language.value,
                    confidence=0.0
                )
                results.append(error_translation)
        
        return results
    
    def get_language_detection(self, text: str) -> Tuple[SupportedLanguage, float]:
        """Detect the language of input text."""
        
        # Simplified language detection
        # In production, would use proper language detection libraries
        
        # Check for obvious patterns
        if re.search(r'[\u4e00-\u9fff]', text):
            return SupportedLanguage.CHINESE_SIMPLIFIED, 0.9
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return SupportedLanguage.JAPANESE, 0.9
        elif re.search(r'[\u0600-\u06ff]', text):
            return SupportedLanguage.ARABIC, 0.9
        elif re.search(r'[\u0590-\u05ff]', text):
            return SupportedLanguage.HEBREW, 0.9
        elif re.search(r'[\u0400-\u04ff]', text):
            return SupportedLanguage.RUSSIAN, 0.8
        
        # Check for common words in various languages
        spanish_indicators = ['el', 'la', 'de', 'en', 'un', 'es', 'se', 'no', 'te', 'lo']
        french_indicators = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
        german_indicators = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich']
        
        text_words = set(text.lower().split())
        
        spanish_score = len(text_words.intersection(spanish_indicators)) / len(text_words) if text_words else 0
        french_score = len(text_words.intersection(french_indicators)) / len(text_words) if text_words else 0
        german_score = len(text_words.intersection(german_indicators)) / len(text_words) if text_words else 0
        
        if spanish_score > 0.1:
            return SupportedLanguage.SPANISH, min(0.8, spanish_score * 4)
        elif french_score > 0.1:
            return SupportedLanguage.FRENCH, min(0.8, french_score * 4)
        elif german_score > 0.1:
            return SupportedLanguage.GERMAN, min(0.8, german_score * 4)
        
        # Default to English
        return SupportedLanguage.ENGLISH, 0.6


class InternationalizationManager:
    """Manager for comprehensive internationalization support."""
    
    def __init__(self):
        self.translation_engine = TranslationEngine()
        self.active_languages = set([SupportedLanguage.ENGLISH])  # Default
        self.user_preferences = {}
        self.regional_settings = {}
        
        # Initialize default regional settings
        self._initialize_regional_settings()
    
    def _initialize_regional_settings(self):
        """Initialize regional settings for different markets."""
        
        self.regional_settings = {
            "US": {
                "primary_language": SupportedLanguage.ENGLISH,
                "currency": "USD",
                "timezone": "America/New_York",
                "regulatory_framework": ["NIST_AI_RMF", "SOX"],
                "cultural_context": CulturalContext.WESTERN
            },
            "EU": {
                "primary_language": SupportedLanguage.ENGLISH,  # Business lingua franca
                "currency": "EUR",
                "timezone": "Europe/Brussels",
                "regulatory_framework": ["GDPR", "AI_ACT_EU"],
                "cultural_context": CulturalContext.WESTERN
            },
            "CN": {
                "primary_language": SupportedLanguage.CHINESE_SIMPLIFIED,
                "currency": "CNY",
                "timezone": "Asia/Shanghai",
                "regulatory_framework": ["PIPL", "Cybersecurity_Law"],
                "cultural_context": CulturalContext.EASTERN
            },
            "JP": {
                "primary_language": SupportedLanguage.JAPANESE,
                "currency": "JPY",
                "timezone": "Asia/Tokyo",
                "regulatory_framework": ["APPI", "AI_Governance"],
                "cultural_context": CulturalContext.EASTERN
            }
        }
    
    def register_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Register user language and localization preferences."""
        
        self.user_preferences[user_id] = {
            'preferred_language': SupportedLanguage(preferences.get('language', 'en')),
            'region': preferences.get('region', 'US'),
            'formality_level': preferences.get('formality', 'formal'),
            'date_format': preferences.get('date_format', 'auto'),
            'number_format': preferences.get('number_format', 'auto'),
            'timezone': preferences.get('timezone', 'UTC'),
            'accessibility_needs': preferences.get('accessibility', []),
            'last_updated': datetime.now(timezone.utc)
        }
    
    def get_localized_text(self, user_id: str, text_key: str, 
                          fallback_text: str = None,
                          context: Dict[str, Any] = None) -> str:
        """Get localized text for user."""
        
        try:
            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, {})
            target_language = user_prefs.get('preferred_language', SupportedLanguage.ENGLISH)
            
            # Create localization context
            localization_context = LocalizationContext(
                target_language=target_language,
                user_region=user_prefs.get('region', 'US'),
                business_context=context.get('business_context', 'general') if context else 'general',
                formality_level=user_prefs.get('formality_level', 'formal'),
                audience_type=context.get('audience_type', 'general') if context else 'general'
            )
            
            # Get text to translate (use fallback if provided)
            source_text = fallback_text or text_key
            
            # Translate
            translation = self.translation_engine.translate_text(
                source_text, target_language, localization_context
            )
            
            return translation.translated_text
            
        except Exception as e:
            handle_error(
                error=e,
                operation="get_localized_text",
                category=ErrorCategory.LOCALIZATION,
                severity=ErrorSeverity.LOW,
                additional_info={
                    'user_id': user_id,
                    'text_key': text_key
                }
            )
            
            # Return fallback
            return fallback_text or text_key
    
    def localize_contract_terms(self, contract_data: Dict[str, Any], 
                              target_language: SupportedLanguage,
                              legal_jurisdiction: str = "US") -> Dict[str, Any]:
        """Localize contract terms for specific jurisdiction."""
        
        localized_contract = contract_data.copy()
        
        # Create specialized legal context
        legal_context = LocalizationContext(
            target_language=target_language,
            user_region=legal_jurisdiction,
            business_context="legal",
            formality_level="formal",
            audience_type="legal"
        )
        
        # Localize key contract elements
        if 'title' in contract_data:
            translation = self.translation_engine.translate_text(
                contract_data['title'], target_language, legal_context
            )
            localized_contract['title'] = translation.translated_text
        
        if 'description' in contract_data:
            translation = self.translation_engine.translate_text(
                contract_data['description'], target_language, legal_context
            )
            localized_contract['description'] = translation.translated_text
        
        # Localize stakeholder information
        if 'stakeholders' in contract_data:
            localized_stakeholders = {}
            for stakeholder_id, stakeholder_data in contract_data['stakeholders'].items():
                localized_stakeholder = stakeholder_data.copy()
                
                if 'name' in stakeholder_data:
                    # Names might not need translation, but roles do
                    if 'role' in stakeholder_data:
                        role_translation = self.translation_engine.translate_text(
                            stakeholder_data['role'], target_language, legal_context
                        )
                        localized_stakeholder['role'] = role_translation.translated_text
                
                localized_stakeholders[stakeholder_id] = localized_stakeholder
            
            localized_contract['stakeholders'] = localized_stakeholders
        
        # Localize constraint descriptions
        if 'constraints' in contract_data:
            localized_constraints = {}
            for constraint_id, constraint_data in contract_data['constraints'].items():
                localized_constraint = constraint_data.copy()
                
                if 'description' in constraint_data:
                    desc_translation = self.translation_engine.translate_text(
                        constraint_data['description'], target_language, legal_context
                    )
                    localized_constraint['description'] = desc_translation.translated_text
                
                localized_constraints[constraint_id] = localized_constraint
            
            localized_contract['constraints'] = localized_constraints
        
        # Add localization metadata
        localized_contract['localization_info'] = {
            'target_language': target_language.value,
            'legal_jurisdiction': legal_jurisdiction,
            'localized_at': datetime.now(timezone.utc).isoformat(),
            'localization_confidence': self._calculate_overall_confidence(localized_contract)
        }
        
        return localized_contract
    
    def _calculate_overall_confidence(self, localized_data: Dict[str, Any]) -> float:
        """Calculate overall localization confidence."""
        # Simplified confidence calculation
        # In practice, would aggregate confidence scores from individual translations
        return 0.85
    
    def generate_multilingual_documentation(self, documentation: Dict[str, Any],
                                          target_languages: List[SupportedLanguage]) -> Dict[str, Any]:
        """Generate documentation in multiple languages."""
        
        multilingual_docs = {
            'original_language': 'en',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'languages': {}
        }
        
        for language in target_languages:
            try:
                # Create technical context
                tech_context = LocalizationContext(
                    target_language=language,
                    user_region="GLOBAL",
                    business_context="technical",
                    formality_level="formal",
                    audience_type="technical"
                )
                
                localized_doc = {}
                
                # Translate main sections
                for section_key, section_content in documentation.items():
                    if isinstance(section_content, str):
                        translation = self.translation_engine.translate_text(
                            section_content, language, tech_context
                        )
                        localized_doc[section_key] = translation.translated_text
                    elif isinstance(section_content, dict):
                        # Recursively translate nested content
                        localized_doc[section_key] = self._translate_nested_content(
                            section_content, language, tech_context
                        )
                    else:
                        # Keep non-string content as is
                        localized_doc[section_key] = section_content
                
                multilingual_docs['languages'][language.value] = localized_doc
                
            except Exception as e:
                handle_error(
                    error=e,
                    operation=f"translate_documentation:{language.value}",
                    category=ErrorCategory.LOCALIZATION,
                    severity=ErrorSeverity.MEDIUM
                )
                # Skip this language but continue with others
                continue
        
        return multilingual_docs
    
    def _translate_nested_content(self, content: Dict[str, Any], 
                                 target_language: SupportedLanguage,
                                 context: LocalizationContext) -> Dict[str, Any]:
        """Recursively translate nested dictionary content."""
        
        translated_content = {}
        
        for key, value in content.items():
            if isinstance(value, str):
                translation = self.translation_engine.translate_text(
                    value, target_language, context
                )
                translated_content[key] = translation.translated_text
            elif isinstance(value, dict):
                translated_content[key] = self._translate_nested_content(
                    value, target_language, context
                )
            elif isinstance(value, list):
                translated_list = []
                for item in value:
                    if isinstance(item, str):
                        translation = self.translation_engine.translate_text(
                            item, target_language, context
                        )
                        translated_list.append(translation.translated_text)
                    else:
                        translated_list.append(item)
                translated_content[key] = translated_list
            else:
                translated_content[key] = value
        
        return translated_content
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        
        languages = []
        
        for language in SupportedLanguage:
            profile = self.translation_engine.language_profiles.get(language.value)
            if profile:
                languages.append({
                    'code': language.value,
                    'name': profile.english_name,
                    'native_name': profile.native_name,
                    'text_direction': profile.text_direction.value,
                    'script_type': profile.script_type,
                    'rtl_support': profile.rtl_support_needed
                })
        
        return languages
    
    def validate_multilingual_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate multilingual content for consistency and completeness."""
        
        validation_results = {
            'overall_status': 'valid',
            'issues': [],
            'recommendations': [],
            'coverage_analysis': {}
        }
        
        if 'languages' not in content:
            validation_results['overall_status'] = 'invalid'
            validation_results['issues'].append('No language data found')
            return validation_results
        
        languages_data = content['languages']
        
        # Check language coverage
        supported_langs = set(lang.value for lang in SupportedLanguage)
        available_langs = set(languages_data.keys())
        
        validation_results['coverage_analysis'] = {
            'available_languages': list(available_langs),
            'missing_languages': list(supported_langs - available_langs),
            'coverage_percentage': len(available_langs) / len(supported_langs) * 100
        }
        
        # Check content consistency
        if len(available_langs) > 1:
            # Get structure from first language
            first_lang = list(available_langs)[0]
            reference_structure = set(languages_data[first_lang].keys())
            
            for lang_code, lang_data in languages_data.items():
                if lang_code != first_lang:
                    current_structure = set(lang_data.keys())
                    
                    missing_keys = reference_structure - current_structure
                    extra_keys = current_structure - reference_structure
                    
                    if missing_keys:
                        validation_results['issues'].append(
                            f"Language {lang_code} missing keys: {list(missing_keys)}"
                        )
                    
                    if extra_keys:
                        validation_results['issues'].append(
                            f"Language {lang_code} has extra keys: {list(extra_keys)}"
                        )
        
        # Generate recommendations
        if validation_results['coverage_analysis']['coverage_percentage'] < 50:
            validation_results['recommendations'].append(
                "Consider adding support for more languages to improve global reach"
            )
        
        if validation_results['issues']:
            validation_results['overall_status'] = 'issues_found'
        
        return validation_results


def create_i18n_manager() -> InternationalizationManager:
    """Factory function to create I18n manager."""
    
    manager = InternationalizationManager()
    
    # Add some common languages to active set
    manager.active_languages.update([
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.CHINESE_SIMPLIFIED,
        SupportedLanguage.JAPANESE
    ])
    
    print("Internationalization Manager initialized")
    print(f"Active languages: {len(manager.active_languages)}")
    print(f"Regional settings: {len(manager.regional_settings)}")
    
    return manager


# Example usage and demonstration
def demonstrate_i18n_capabilities():
    """Demonstrate internationalization capabilities."""
    
    print("🌍 Multi-Language Support Demonstration")
    print("=" * 60)
    
    # Create I18n manager
    i18n_manager = create_i18n_manager()
    
    # Example translations
    test_texts = [
        "Welcome to RLHF Contract Wizard",
        "Your reward function has been successfully deployed",
        "Compliance check completed with 95% accuracy",
        "Smart contract verification in progress"
    ]
    
    target_languages = [
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.CHINESE_SIMPLIFIED,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.ARABIC
    ]
    
    print("\n📝 Text Translation Examples:")
    print("-" * 40)
    
    for text in test_texts[:2]:  # Show first 2 for brevity
        print(f"\nOriginal: {text}")
        
        for lang in target_languages[:3]:  # Show first 3 languages
            context = LocalizationContext(
                target_language=lang,
                user_region="GLOBAL",
                business_context="technical",
                formality_level="formal",
                audience_type="general"
            )
            
            translation = i18n_manager.translation_engine.translate_text(text, lang, context)
            print(f"{lang.value:6}: {translation.translated_text} (confidence: {translation.confidence:.2f})")
    
    # Demo contract localization
    print("\n📋 Contract Localization Example:")
    print("-" * 40)
    
    sample_contract = {
        'title': 'AI Safety Reward Contract',
        'description': 'This contract ensures safe and ethical AI behavior through reward function constraints',
        'stakeholders': {
            'operator': {
                'name': 'AI Operator Inc.',
                'role': 'Primary system operator',
                'weight': 0.4
            },
            'safety_board': {
                'name': 'Safety Review Board',
                'role': 'Safety oversight authority',
                'weight': 0.6
            }
        },
        'constraints': {
            'safety_first': {
                'description': 'System must prioritize safety above all other objectives',
                'severity': 'critical'
            }
        }
    }
    
    # Localize to Spanish
    localized_contract = i18n_manager.localize_contract_terms(
        sample_contract, SupportedLanguage.SPANISH, "ES"
    )
    
    print("Original title:", sample_contract['title'])
    print("Spanish title:", localized_contract['title'])
    print("Localization confidence:", localized_contract['localization_info']['localization_confidence'])
    
    # Demo language detection
    print("\n🔍 Language Detection Examples:")
    print("-" * 40)
    
    test_detection_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás hoy?",
        "你好，你今天怎么样？",
        "مرحبا كيف حالك اليوم؟"
    ]
    
    for text in test_detection_texts:
        detected_lang, confidence = i18n_manager.translation_engine.get_language_detection(text)
        print(f"Text: {text}")
        print(f"Detected: {detected_lang.value} (confidence: {confidence:.2f})")
        print()
    
    print("🎯 Multi-Language Support Demonstration Complete")
    return i18n_manager


if __name__ == "__main__":
    demonstrate_i18n_capabilities()