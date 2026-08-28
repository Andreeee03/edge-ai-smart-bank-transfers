import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const EdgeAiApp());
}

class EdgeAiApp extends StatelessWidget {
  const EdgeAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Smart Bank Transfer',
      theme: ThemeData(
        useMaterial3: true,
      ),
      home: const TransferPage(),
    );
  }
}

enum ActivityType {
  generation,
  completion,
  normalization,
}

class TransferPage extends StatefulWidget {
  const TransferPage({super.key});

  @override
  State<TransferPage> createState() => _TransferPageState();
}

class _TransferPageState extends State<TransferPage> {
  static const platform = MethodChannel('edge_ai/native');

  ActivityType _activity = ActivityType.generation;

  final _ibanController = TextEditingController();
  final _categoryController = TextEditingController();
  final _beneficiaryController = TextEditingController();
  final _amountController = TextEditingController();
  final _currencyController = TextEditingController(text: 'EUR');
  final _referencePeriodController = TextEditingController();
  final _inputTextController = TextEditingController();

  final _suggestion1Controller = TextEditingController();
  final _suggestion2Controller = TextEditingController();

  // Sempre disponibile anche senza AI.
  final _finalDescriptionController = TextEditingController();

  bool _busy = false;
  bool _aiReady = false;

  String _status = 'Ready';

  String _clean(String value) => value.trim();

  String _formatAmount(String value) {
    final normalized = value.trim().replaceAll(',', '.');

    final parsed = double.tryParse(normalized);

    if (parsed == null) {
      return normalized;
    }

    return parsed
        .toStringAsFixed(2)
        .replaceFirst(RegExp(r'0+$'), '')
        .replaceFirst(RegExp(r'\.$'), '');
  }

  String _buildPrompt() {
    final category =
        _clean(_categoryController.text).toUpperCase();

    final beneficiary =
        _clean(_beneficiaryController.text);

    final amount =
        _formatAmount(_amountController.text);

    final currency =
        _clean(_currencyController.text).toUpperCase();

    final referencePeriod =
        _clean(_referencePeriodController.text);

    final inputText =
        _clean(_inputTextController.text);

    final parts = <String>[];

    switch (_activity) {
      case ActivityType.generation:
        parts.add(
          'Generate exactly two concise and natural '
          'bank-transfer descriptions using only the '
          'information provided.',
        );

        parts.add(
          'Return two alternative descriptions without '
          'adding unsupported information.',
        );

        break;

      case ActivityType.completion:
        parts.add(
          'Complete the following partially written '
          'bank-transfer description.',
        );

        parts.add(
          'Generate exactly two concise and natural '
          'completed alternatives using only the '
          'information provided.',
        );

        break;

      case ActivityType.normalization:
        parts.add(
          'Normalize the following bank-transfer '
          'description by making it clear, concise '
          'and natural.',
        );

        parts.add(
          'Generate exactly two alternative normalized '
          'descriptions while preserving the original '
          'meaning and without adding unsupported '
          'information.',
        );

        break;
    }

    parts.add('Category: $category');
    parts.add('Beneficiary: $beneficiary');
    parts.add('Amount: $amount $currency');

    if (referencePeriod.isNotEmpty) {
      parts.add(
        'Reference period: $referencePeriod',
      );
    }

    if (_activity == ActivityType.completion) {
      parts.add(
        'Partial description: $inputText',
      );
    }

    if (_activity == ActivityType.normalization) {
      parts.add(
        'Original description: $inputText',
      );
    }

    /*
     * IMPORTANT:
     *
     * IBAN NON viene inserito nel prompt.
     *
     * Il modello non lo ha visto durante il fine-tuning.
     *
     * Boundary identico al training:
     * prompt.rstrip("\r\n") + "\n\n"
     */
    final prompt =
        parts.join('\n').replaceFirst(
              RegExp(r'[\r\n]+$'),
              '',
            );

    return '$prompt\n\n';
  }

  String? _validateForm() {
    if (_categoryController.text.trim().isEmpty) {
      return 'Insert the operation category.';
    }

    if (_beneficiaryController.text.trim().isEmpty) {
      return 'Insert the beneficiary.';
    }

    if (_amountController.text.trim().isEmpty) {
      return 'Insert the amount.';
    }

    if (_currencyController.text.trim().isEmpty) {
      return 'Insert the currency.';
    }

    if ((_activity == ActivityType.completion ||
            _activity == ActivityType.normalization) &&
        _inputTextController.text.trim().isEmpty) {
      return _activity == ActivityType.completion
          ? 'Insert the partial description.'
          : 'Insert the description to normalize.';
    }

    return null;
  }

  Future<void> _ensureAiReady() async {
    if (_aiReady) {
      return;
    }

    setState(() {
      _status = 'Loading local AI model...';
    });

    await platform.invokeMethod<String>(
      'loadModel',
    );

    setState(() {
      _status = 'Preparing local inference...';
    });

    await platform.invokeMethod<String>(
      'createContext',
    );

    _aiReady = true;
  }

  String _extractModelText(String raw) {
    const marker = 'Output:\n';

    final index = raw.indexOf(marker);

    if (index >= 0) {
      return raw
          .substring(index + marker.length)
          .trim();
    }

    return raw.trim();
  }

  List<String> _parseSuggestions(
    String output,
  ) {
    String? first;
    String? second;

    final lines = output.split('\n');

    for (final line in lines) {
      final trimmed = line.trim();

      final firstMatch =
          RegExp(r'^1\.\s*(.+)$').firstMatch(trimmed);

      final secondMatch =
          RegExp(r'^2\.\s*(.+)$').firstMatch(trimmed);

      if (firstMatch != null) {
        first = firstMatch.group(1)?.trim();
      }

      if (secondMatch != null) {
        second = secondMatch.group(1)?.trim();
      }
    }

    return [
      if (first != null && first.isNotEmpty)
        first,
      if (second != null && second.isNotEmpty)
        second,
    ];
  }

  Future<void> _generate() async {
    final error = _validateForm();

    if (error != null) {
      setState(() {
        _status = error;
      });

      return;
    }

    setState(() {
      _busy = true;
      _status = 'Preparing generation...';

      _suggestion1Controller.clear();
      _suggestion2Controller.clear();
    });

    try {
      await _ensureAiReady();

      final prompt = _buildPrompt();

      setState(() {
        _status =
            'Generating locally on this device...';
      });

      final result =
          await platform.invokeMethod<String>(
        'generate',
        {
          'prompt': prompt,
        },
      );

      final modelText =
          _extractModelText(result ?? '');

      final suggestions =
          _parseSuggestions(modelText);

      if (suggestions.length < 2) {
        setState(() {
          _status =
              'The AI did not return two valid suggestions.';
        });

        return;
      }

      _suggestion1Controller.text =
          suggestions[0];

      _suggestion2Controller.text =
          suggestions[1];

      setState(() {
        _status =
            'Two suggestions generated locally';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status =
            'AI unavailable. You can still enter the description manually.';
      });

      debugPrint(
        'Local inference error: ${e.code}',
      );
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  void _useSuggestion(
    TextEditingController controller,
  ) {
    _finalDescriptionController.text =
        controller.text.trim();

    setState(() {
      _status =
          'AI suggestion selected. You can still edit it.';
    });
  }

  void _rejectSuggestions() {
    _suggestion1Controller.clear();
    _suggestion2Controller.clear();

    setState(() {
      _status =
          'Suggestions rejected. Manual entry is still available.';
    });
  }

  String _activityLabel(
    ActivityType value,
  ) {
    switch (value) {
      case ActivityType.generation:
        return 'Generation';

      case ActivityType.completion:
        return 'Completion';

      case ActivityType.normalization:
        return 'Normalization';
    }
  }

  @override
  void dispose() {
    _ibanController.dispose();
    _categoryController.dispose();
    _beneficiaryController.dispose();
    _amountController.dispose();
    _currencyController.dispose();
    _referencePeriodController.dispose();
    _inputTextController.dispose();
    _suggestion1Controller.dispose();
    _suggestion2Controller.dispose();
    _finalDescriptionController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final needsInputText =
        _activity == ActivityType.completion ||
        _activity == ActivityType.normalization;

    final hasSuggestions =
        _suggestion1Controller.text.isNotEmpty ||
        _suggestion2Controller.text.isNotEmpty;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Smart Bank Transfer',
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment:
              CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding:
                    const EdgeInsets.all(12),
                child: Row(
                  children: [
                    const Icon(
                      Icons.smartphone,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'On-device AI • '
                        'Your transfer data is processed locally',
                        style:
                            Theme.of(context)
                                .textTheme
                                .bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            TextField(
              controller: _ibanController,
              decoration:
                  const InputDecoration(
                labelText: 'IBAN',
                helperText:
                    'Transfer field — not sent to the AI model',
                border:
                    OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            DropdownButtonFormField<ActivityType>(
              initialValue: _activity,
              decoration:
                  const InputDecoration(
                labelText: 'AI assistance',
                border:
                    OutlineInputBorder(),
              ),
              items: ActivityType.values
                  .map(
                    (activity) =>
                        DropdownMenuItem(
                      value: activity,
                      child: Text(
                        _activityLabel(
                          activity,
                        ),
                      ),
                    ),
                  )
                  .toList(),
              onChanged: _busy
                  ? null
                  : (value) {
                      if (value != null) {
                        setState(() {
                          _activity = value;

                          _suggestion1Controller
                              .clear();

                          _suggestion2Controller
                              .clear();
                        });
                      }
                    },
            ),

            const SizedBox(height: 16),

            TextField(
              controller:
                  _categoryController,
              decoration:
                  const InputDecoration(
                labelText: 'Category',
                border:
                    OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            TextField(
              controller:
                  _beneficiaryController,
              decoration:
                  const InputDecoration(
                labelText: 'Beneficiary',
                border:
                    OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: TextField(
                    controller:
                        _amountController,
                    keyboardType:
                        const TextInputType
                            .numberWithOptions(
                      decimal: true,
                    ),
                    decoration:
                        const InputDecoration(
                      labelText: 'Amount',
                      border:
                          OutlineInputBorder(),
                    ),
                  ),
                ),

                const SizedBox(width: 12),

                Expanded(
                  child: TextField(
                    controller:
                        _currencyController,
                    decoration:
                        const InputDecoration(
                      labelText: 'Currency',
                      border:
                          OutlineInputBorder(),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            TextField(
              controller:
                  _referencePeriodController,
              decoration:
                  const InputDecoration(
                labelText:
                    'Reference period (optional)',
                border:
                    OutlineInputBorder(),
              ),
            ),

            if (needsInputText) ...[
              const SizedBox(height: 16),

              TextField(
                controller:
                    _inputTextController,
                maxLines: 3,
                decoration:
                    InputDecoration(
                  labelText:
                      _activity ==
                              ActivityType
                                  .completion
                          ? 'Partial description'
                          : 'Original description',
                  border:
                      const OutlineInputBorder(),
                ),
              ),
            ],

            const SizedBox(height: 22),

            FilledButton.icon(
              onPressed:
                  _busy ? null : _generate,
              icon: const Icon(
                Icons.auto_awesome,
              ),
              label: Text(
                _busy
                    ? 'Generating...'
                    : hasSuggestions
                        ? 'Regenerate suggestions'
                        : 'Generate suggestions',
              ),
            ),

            if (_busy) ...[
              const SizedBox(height: 14),
              const LinearProgressIndicator(),
            ],

            const SizedBox(height: 18),

            Text(
              _status,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall,
            ),

            if (_suggestion1Controller
                    .text.isNotEmpty) ...[
              const SizedBox(height: 24),

              const Text(
                'AI suggestion 1',
                style: TextStyle(
                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              const SizedBox(height: 8),

              TextField(
                controller:
                    _suggestion1Controller,
                maxLines: 2,
                decoration:
                    const InputDecoration(
                  border:
                      OutlineInputBorder(),
                ),
              ),

              const SizedBox(height: 8),

              OutlinedButton(
                onPressed: () =>
                    _useSuggestion(
                  _suggestion1Controller,
                ),
                child:
                    const Text(
                  'Use suggestion 1',
                ),
              ),
            ],

            if (_suggestion2Controller
                    .text.isNotEmpty) ...[
              const SizedBox(height: 18),

              const Text(
                'AI suggestion 2',
                style: TextStyle(
                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              const SizedBox(height: 8),

              TextField(
                controller:
                    _suggestion2Controller,
                maxLines: 2,
                decoration:
                    const InputDecoration(
                  border:
                      OutlineInputBorder(),
                ),
              ),

              const SizedBox(height: 8),

              OutlinedButton(
                onPressed: () =>
                    _useSuggestion(
                  _suggestion2Controller,
                ),
                child:
                    const Text(
                  'Use suggestion 2',
                ),
              ),
            ],

            if (hasSuggestions) ...[
              const SizedBox(height: 10),

              TextButton(
                onPressed:
                    _rejectSuggestions,
                child:
                    const Text(
                  'Reject AI suggestions',
                ),
              ),
            ],

            const SizedBox(height: 24),

            const Divider(),

            const SizedBox(height: 16),

            const Text(
              'Final transfer description',
              style: TextStyle(
                fontWeight:
                    FontWeight.bold,
              ),
            ),

            const SizedBox(height: 8),

            TextField(
              controller:
                  _finalDescriptionController,
              maxLines: 3,
              decoration:
                  const InputDecoration(
                hintText:
                    'Write manually or select an AI suggestion',
                border:
                    OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 10),

            const Text(
              'The transfer is simulated. '
              'No payment will be executed.',
              style: TextStyle(
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
