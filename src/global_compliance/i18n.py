"""
Internationalization (i18n) support for RLHF Contract Wizard.

Provides multi-language support for global deployment, supporting:
en, es, fr, de, ja, zh as required by the SDLC specification.
"""

import json
import os
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    translation_cache_size: int = 1000
    auto_detect_language: bool = True


class InternationalizationManager:
    """
    Manages internationalization and localization for the system.
    
    Provides translation services, cultural formatting, and locale-specific
    business logic for global RLHF contract deployment.
    """
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = self.config.default_language
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        # Base translations for system messages
        self.translations = {
            "en": {
                "system.startup": "RLHF Contract Wizard starting up",
                "system.shutdown": "System shutting down gracefully",
                "contract.created": "Contract created successfully",
                "contract.validated": "Contract validation completed",
                "contract.deployed": "Contract deployed to blockchain",
                "error.validation_failed": "Validation failed",
                "error.network_error": "Network connection error",
                "error.permission_denied": "Permission denied",
                "quality.gates_passed": "All quality gates passed",
                "quality.gates_failed": "Quality gates failed",
                "deployment.ready": "System ready for deployment",
                "deployment.not_ready": "System not ready for deployment",
                "stakeholder.notification": "Stakeholder notification sent",
                "compliance.gdpr": "GDPR compliance verified",
                "compliance.ccpa": "CCPA compliance verified",
                "compliance.pdpa": "PDPA compliance verified"
            },
            "es": {
                "system.startup": "Asistente de Contratos RLHF iniciándose",
                "system.shutdown": "Sistema cerrándose correctamente",
                "contract.created": "Contrato creado exitosamente",
                "contract.validated": "Validación de contrato completada",
                "contract.deployed": "Contrato desplegado en blockchain",
                "error.validation_failed": "La validación falló",
                "error.network_error": "Error de conexión de red",
                "error.permission_denied": "Permiso denegado",
                "quality.gates_passed": "Todas las compuertas de calidad pasaron",
                "quality.gates_failed": "Las compuertas de calidad fallaron",
                "deployment.ready": "Sistema listo para despliegue",
                "deployment.not_ready": "Sistema no listo para despliegue",
                "stakeholder.notification": "Notificación de stakeholder enviada",
                "compliance.gdpr": "Cumplimiento GDPR verificado",
                "compliance.ccpa": "Cumplimiento CCPA verificado",
                "compliance.pdpa": "Cumplimiento PDPA verificado"
            },
            "fr": {
                "system.startup": "Assistant de Contrats RLHF en cours de démarrage",
                "system.shutdown": "Arrêt gracieux du système",
                "contract.created": "Contrat créé avec succès",
                "contract.validated": "Validation du contrat terminée",
                "contract.deployed": "Contrat déployé sur blockchain",
                "error.validation_failed": "La validation a échoué",
                "error.network_error": "Erreur de connexion réseau",
                "error.permission_denied": "Permission refusée",
                "quality.gates_passed": "Toutes les portes de qualité sont passées",
                "quality.gates_failed": "Les portes de qualité ont échoué",
                "deployment.ready": "Système prêt pour le déploiement",
                "deployment.not_ready": "Système non prêt pour le déploiement",
                "stakeholder.notification": "Notification des parties prenantes envoyée",
                "compliance.gdpr": "Conformité GDPR vérifiée",
                "compliance.ccpa": "Conformité CCPA vérifiée",
                "compliance.pdpa": "Conformité PDPA vérifiée"
            },
            "de": {
                "system.startup": "RLHF Vertrags-Assistent startet",
                "system.shutdown": "System fährt ordnungsgemäß herunter",
                "contract.created": "Vertrag erfolgreich erstellt",
                "contract.validated": "Vertragsvalidierung abgeschlossen",
                "contract.deployed": "Vertrag in Blockchain bereitgestellt",
                "error.validation_failed": "Validierung fehlgeschlagen",
                "error.network_error": "Netzwerkverbindungsfehler",
                "error.permission_denied": "Berechtigung verweigert",
                "quality.gates_passed": "Alle Qualitätstüren bestanden",
                "quality.gates_failed": "Qualitätstüren fehlgeschlagen",
                "deployment.ready": "System bereit für Bereitstellung",
                "deployment.not_ready": "System nicht bereit für Bereitstellung",
                "stakeholder.notification": "Stakeholder-Benachrichtigung gesendet",
                "compliance.gdpr": "GDPR-Konformität überprüft",
                "compliance.ccpa": "CCPA-Konformität überprüft",
                "compliance.pdpa": "PDPA-Konformität überprüft"
            },
            "ja": {
                "system.startup": "RLHF契約ウィザードを起動中",
                "system.shutdown": "システムを正常にシャットダウン中",
                "contract.created": "契約が正常に作成されました",
                "contract.validated": "契約の検証が完了しました",
                "contract.deployed": "契約がブロックチェーンにデプロイされました",
                "error.validation_failed": "検証に失敗しました",
                "error.network_error": "ネットワーク接続エラー",
                "error.permission_denied": "アクセス拒否",
                "quality.gates_passed": "すべての品質ゲートが通過しました",
                "quality.gates_failed": "品質ゲートが失敗しました",
                "deployment.ready": "システムはデプロイ準備完了",
                "deployment.not_ready": "システムはデプロイ準備未完了",
                "stakeholder.notification": "ステークホルダー通知を送信しました",
                "compliance.gdpr": "GDPR準拠を確認しました",
                "compliance.ccpa": "CCPA準拠を確認しました",
                "compliance.pdpa": "PDPA準拠を確認しました"
            },
            "zh": {
                "system.startup": "RLHF合约向导启动中",
                "system.shutdown": "系统正常关闭中",
                "contract.created": "合约创建成功",
                "contract.validated": "合约验证完成",
                "contract.deployed": "合约已部署到区块链",
                "error.validation_failed": "验证失败",
                "error.network_error": "网络连接错误",
                "error.permission_denied": "权限被拒绝",
                "quality.gates_passed": "所有质量门已通过",
                "quality.gates_failed": "质量门失败",
                "deployment.ready": "系统已准备好部署",
                "deployment.not_ready": "系统未准备好部署",
                "stakeholder.notification": "利益相关者通知已发送",
                "compliance.gdpr": "GDPR合规性已验证",
                "compliance.ccpa": "CCPA合规性已验证",
                "compliance.pdpa": "PDPA合规性已验证"
            }
        }
    
    def set_language(self, language: SupportedLanguage):
        """Set the current language for translations."""
        self.current_language = language
    
    def translate(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """
        Translate a message key to the specified or current language.
        
        Args:
            key: The translation key
            language: Optional language override
            **kwargs: Format arguments for the translation
        
        Returns:
            Translated message string
        """
        target_language = language or self.current_language
        lang_code = target_language.value
        
        # Get translation or fall back to English
        if lang_code in self.translations:
            translation = self.translations[lang_code].get(key)
            if translation:
                return translation.format(**kwargs) if kwargs else translation
        
        # Fallback to English
        english_translation = self.translations["en"].get(key, key)
        return english_translation.format(**kwargs) if kwargs else english_translation
    
    def get_supported_languages(self) -> list[SupportedLanguage]:
        """Get list of supported languages."""
        return list(SupportedLanguage)
    
    def detect_language_from_header(self, accept_language: str) -> SupportedLanguage:
        """
        Detect preferred language from HTTP Accept-Language header.
        
        Args:
            accept_language: Accept-Language header value
        
        Returns:
            Best matching supported language
        """
        if not accept_language:
            return self.config.default_language
        
        # Parse Accept-Language header (simplified)
        languages = []
        for lang_entry in accept_language.split(','):
            lang_entry = lang_entry.strip()
            if ';q=' in lang_entry:
                lang, quality = lang_entry.split(';q=')
                quality = float(quality)
            else:
                lang, quality = lang_entry, 1.0
            
            # Extract primary language code
            primary_lang = lang.split('-')[0].lower()
            languages.append((primary_lang, quality))
        
        # Sort by quality score
        languages.sort(key=lambda x: x[1], reverse=True)
        
        # Find best match
        for lang_code, _ in languages:
            for supported_lang in SupportedLanguage:
                if supported_lang.value == lang_code:
                    return supported_lang
        
        return self.config.default_language
    
    def format_currency(self, amount: float, currency: str = "USD", language: Optional[SupportedLanguage] = None) -> str:
        """
        Format currency according to locale conventions.
        
        Args:
            amount: Currency amount
            currency: Currency code (USD, EUR, JPY, etc.)
            language: Target language for formatting
        
        Returns:
            Formatted currency string
        """
        target_language = language or self.current_language
        
        # Simplified currency formatting per locale
        formatting_rules = {
            SupportedLanguage.ENGLISH: "${:,.2f}",
            SupportedLanguage.SPANISH: "{:,.2f} {}",
            SupportedLanguage.FRENCH: "{:,.2f} {}",
            SupportedLanguage.GERMAN: "{:,.2f} {}",
            SupportedLanguage.JAPANESE: "{}¥{:,.0f}",
            SupportedLanguage.CHINESE: "{}¥{:,.2f}"
        }
        
        format_str = formatting_rules.get(target_language, "${:,.2f}")
        
        if target_language == SupportedLanguage.ENGLISH:
            return format_str.format(amount)
        elif target_language in [SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE]:
            return format_str.format(currency, amount)
        else:
            return format_str.format(amount, currency)
    
    def format_datetime(self, timestamp: float, language: Optional[SupportedLanguage] = None) -> str:
        """
        Format datetime according to locale conventions.
        
        Args:
            timestamp: Unix timestamp
            language: Target language for formatting
        
        Returns:
            Formatted datetime string
        """
        import datetime
        
        target_language = language or self.current_language
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        # Locale-specific datetime formats
        if target_language == SupportedLanguage.ENGLISH:
            return dt.strftime("%m/%d/%Y %I:%M %p")
        elif target_language == SupportedLanguage.SPANISH:
            return dt.strftime("%d/%m/%Y %H:%M")
        elif target_language == SupportedLanguage.FRENCH:
            return dt.strftime("%d/%m/%Y %H:%M")
        elif target_language == SupportedLanguage.GERMAN:
            return dt.strftime("%d.%m.%Y %H:%M")
        elif target_language == SupportedLanguage.JAPANESE:
            return dt.strftime("%Y年%m月%d日 %H:%M")
        elif target_language == SupportedLanguage.CHINESE:
            return dt.strftime("%Y年%m月%d日 %H:%M")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")


# Global i18n manager instance
_i18n_manager = None


def get_i18n_manager() -> InternationalizationManager:
    """Get the global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = InternationalizationManager()
    return _i18n_manager


def translate(key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
    """Convenience function for translations."""
    return get_i18n_manager().translate(key, language, **kwargs)


def set_language(language: SupportedLanguage):
    """Convenience function to set current language."""
    get_i18n_manager().set_language(language)